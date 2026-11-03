# usage: python -I -B maker_fixture.py <onejev> <requests.json> <reference-bfloat16.json> <maker-prompts.json>
# Unit fixture for the Swift prompt/answer tests: requests as JSON text (key order kept), the maker's prompts, its slot
# logits and the answers qev's own answer functions build from them (CPU, deterministic).
import json, sys
sys.path.insert(0, sys.argv[1])
from qev.engine import DecisionEngine  # noqa: F401  (import check only)
from qev.calibrate import softmax
from qev.answers import make_choice, make_noul, make_score
from qev.prompt import render_question
from qev.schema import SystemOneRequest
cases = json.load(open(sys.argv[2])); reference = json.load(open(sys.argv[3])); ref = {c['id']: c for c in reference['cases']}
out = []
for case in cases:
    r = ref[case['id']]
    request = SystemOneRequest(state=case['state'], questions=case['questions'])
    rendered = [render_question(k, q) for k, q in request.questions.items()]
    answers = {}
    for (qid, q), br, logits in zip(request.questions.items(), rendered, r['slot_logits']):
        p = softmax(logits)
        a = make_noul(p[0]) if br.kind == 'noul' else make_choice(br.labels, p) if br.kind == 'choice' else make_score(br.descriptions, p)
        answers[qid] = a.model_dump()
    out.append({'id': case['id'], 'request': json.dumps({'state': case['state'], 'questions': case['questions']}, ensure_ascii=False),
                'prompts': r['prompts'], 'suffixes': [b.suffix for b in rendered], 'slot_logits': r['slot_logits'],
                'answers': json.dumps(answers, ensure_ascii=False, sort_keys=True, separators=(',', ':')), 'prefix_ids': r['prefix_ids'], 'suffix_ids': r['suffix_ids']})
json.dump({'source': 'OneJev qev 81ce62f + elephant-1 serve.patch; slot logits from MPS bfloat16; slot_ids from the maker engine', 'cases': out, 'slot_ids': reference['env']['slot_ids']}, open(sys.argv[4], 'w'), ensure_ascii=False)
