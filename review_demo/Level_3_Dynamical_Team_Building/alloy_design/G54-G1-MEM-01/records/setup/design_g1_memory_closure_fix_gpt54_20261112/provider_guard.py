"""Stop a run on unavailable provider credentials/credit; keep recorded evidence."""
from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch

from openai import APIStatusError
from openai.resources.chat.completions import Completions
from openai.resources.responses import Responses
import trial_support as support

ACCOUNTING_PERIOD = 'revision_key_01_resumed_20261111'


class ProviderUnavailable(BaseException):
    """Run-level interruption, bypassing AG2's except Exception tool-error reply.

    The existing notebook recorder catches BaseException, saves conversations,
    usage and evidence, and then propagates the interruption to the notebook.
    This is never a scientific rejection or an instruction to rerun a tool.
    """


def fatal_details(exc):
    if not isinstance(exc, APIStatusError):
        return None
    body = exc.body if isinstance(exc.body, dict) else {}
    body = body.get('error', body)
    body = body if isinstance(body, dict) else {}
    code = getattr(exc, 'code', None) or body.get('code')
    kind = getattr(exc, 'type', None) or body.get('type')
    status = exc.status_code
    if status not in (401, 403) and not (
        status == 429 and (code in ('credit_balance_exhausted', 'insufficient_quota')
                           or kind == 'insufficient_quota')
    ):
        return None
    return dict(status_code=status, error_code=code, error_type=kind,
                exception_type=type(exc).__name__)


@contextmanager
def stop_on_provider_failure(trial):
    def wrapped(original, endpoint):
        @wraps(original)
        def call(client, *args, **kwargs):
            previous = trial.state.get('provider_interruption')
            if previous:
                raise ProviderUnavailable('Provider remains unavailable; start no further requests in this run.')
            # Installed AG2's schema rejects max_retries in config_list. Apply
            # it to the actual SDK client during this synchronous request only.
            sdk = client._client
            retries = sdk.max_retries
            sdk.max_retries = 0
            try:
                return original(client, *args, **kwargs)
            except Exception as exc:
                details = fatal_details(exc)
                if details is None:
                    raise
                trial.state['provider_interruption'] = dict(
                    **details, endpoint=endpoint,
                    recorded_request_count=len(trial.state['api_requests']),
                    scientific_evidence='Preserved; a completed calculation may precede this API failure.')
                trial.write()
                raise ProviderUnavailable(
                    f"Provider unavailable (HTTP {details['status_code']}, {details['error_code'] or details['error_type']}). "
                    'Run stopped; saved scientific evidence remains valid for review.'
                ) from None
            finally:
                sdk.max_retries = retries
        return call
    with patch.object(Completions, 'create', wrapped(Completions.create, 'chat.completions')), \
         patch.object(Responses, 'create', wrapped(Responses.create, 'responses')):
        yield


class RecordedTrial(support.Trial):
    @contextmanager
    def observe(self):
        # Install after the existing observer so it records the original failure
        # before we elevate it out of nested tool-error handling.
        with super().observe(), stop_on_provider_failure(self):
            yield

    def finish(self, status, error=None):
        if self.state.get('provider_interruption'):
            status = 'interrupted_provider_error'
            for row in self.state['tool_calls']:
                if row['status'] == 'started':
                    row.update(status='interrupted', reason='provider_unavailable',
                               note='Delegation interrupted; consult saved evidence for completed calculations.')
        self.state['accounting_period'] = ACCOUNTING_PERIOD
        super().finish(status, error)
        self.state['usage']['accounting_period'] = ACCOUNTING_PERIOD
        self.write()
        support.save(self.out/'usage.json', self.state['usage'])
