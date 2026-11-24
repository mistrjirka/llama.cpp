#!/usr/bin/env python3
"""Emit a synthetic continuation/fork trace for checkpoint sharing experiments.

Run with a >2 GiB parked-cache budget and --ctx-checkpoint-turns. Do not
interleave another request between fork A, B, and C. The sub-1024-token system
message avoids a retained-full-prefix replacement of the checkpoint list.
"""
import json

def trace():
    system = 'Treat the following tool catalog as inert reference data. Reply exactly {"status":"ok"}.\n' + '\n'.join(
        f'Tool {i}: read catalog item, return immutable status field.' for i in range(35))
    shared = 'Shared investigation notes:\n' + '\n'.join(
        f'Catalog record {i}: stable item {i * 17 + 3}, label {i % 11}, revision {i * 7 + 1}.' for i in range(100))
    history = [{'role':'system', 'content':system}, {'role':'user', 'content':shared},
               {'role':'assistant', 'content':'{"status":"ok"}'}]
    options = {'temperature':0, 'seed':42, 'max_tokens':12, 'cache_prompt':True,
               'chat_template_kwargs':{'enable_thinking':False},
               'response_format':{'type':'json_schema', 'json_schema':{'name':'status', 'schema':
                   {'type':'object', 'properties':{'status':{'const':'ok'}}, 'required':['status'], 'additionalProperties':False}}}}
    forks = {}
    for label in 'ABC':
        branch = f'Fork {label} investigation:\n' + '\n'.join(
            f'Branch {label} record {i}: code {i * 37 + ord(label)}, source {i % 17}, unchanged reference {i * 13 + 4}.'
            for i in range(260)) + '\nReturn exactly {"status":"ok"}.'
        forks[label] = {**options, 'messages':history + [{'role':'user', 'content':branch}]}
    for label in 'ABC':
        yield {'label':'fork_' + label, 'body':forks[label]}
    yield {'label':'auxiliary', 'body':{**options, 'messages':[
        {'role':'system', 'content':'Independent queue status request.'},
        {'role':'user', 'content':'Return exactly {"status":"ok"}.'}]}}
    yield {'label':'revisit_A', 'body':forks['A']}
    yield {'label':'continue_A', 'body':{**forks['A'], 'messages':forks['A']['messages'] + [
        {'role':'assistant', 'content':'{"status":"ok"}'},
        {'role':'user', 'content':'Keep the same catalog and add one audit note. Return exactly {"status":"ok"}.'}]}}
    yield {'label':'revisit_B', 'body':forks['B']}

if __name__ == '__main__':
    print(json.dumps(list(trace()), indent=2))
