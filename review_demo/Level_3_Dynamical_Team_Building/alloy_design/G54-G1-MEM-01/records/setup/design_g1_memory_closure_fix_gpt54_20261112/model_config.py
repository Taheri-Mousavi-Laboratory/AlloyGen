"""Inherited GPT-5.4 request settings for optional inter-round memory tests."""
import json
import os
from . import tools


def reasoning_config(trial, offline=False):
    config = trial.config(offline=offline)
    # The live API rejected medium reasoning + tools on Chat Completions.
    # Use GPT-5.4's default none, retaining the existing framework. Installed
    # AG2's enum does not accept explicit 'none', so omit reasoning_effort.
    # The builder reloads config_list from its env; keep both copies identical.
    config.pop('temperature', None)
    config.pop('max_tokens', None)
    config['timeout'] = 300
    entry = config['config_list'][0]
    assert tools.PROTOCOL['reasoning_effort'] == 'none'
    entry.update(max_completion_tokens=tools.PROTOCOL['max_completion_tokens'],
                 price=[0.0025, 0.015])  # AG2 uses USD/1000; observer includes caching.
    os.environ[trial.config_env] = json.dumps(config['config_list'])
    trial.state['model_comparison'] = dict(
        baseline_run='alloy_design_G1_final_ready_gpt54_20261111_01',
        model=entry['model'], reasoning_effort='none (model default; field omitted)',
        max_completion_tokens=entry['max_completion_tokens'], temperature='omitted',
        timeout_seconds=300, max_retries=0, retry_control='runtime provider guard on actual SDK client',
        retrieval_tokenizer_model=tools.PROTOCOL['retrieval_tokenizer_model'],
        unchanged='model generation settings, physics, overall domain/limits, five roles and team-selection mechanism',
        changed='optional run working memory, generic manager memory instructions and report references',
        scope=('offline scripted integration only' if offline else
               'fresh G1 investigation; no inherited measurements, nomination, verification or source conversation'))
    trial.write()
    return config
