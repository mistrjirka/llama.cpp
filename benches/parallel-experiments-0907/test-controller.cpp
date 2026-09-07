#include "tools/server/server-serving-experiment.h"
#include <cassert>
int main() {
    server_serving_experiment c;
    assert(c.prompt_limit(2048, 16) == 2048);
    c.prefill_budget_ms = 100;
    c.observe_prefill(100, 200000);
    assert(c.prompt_limit(2048, 16) == 66);
    assert(c.prompt_limit(2048, 0) == 2048);
    for (int i=0;i<8;++i) {
        int depth=c.choose_depth(4,3);
        c.observe_depth(4,depth,4, depth==2 ? 400 : 800);
    }
    assert(c.choose_depth(4,3)==2);
    assert(c.choose_depth(1,3)==0);
    assert(c.prompt_limit(16, 4)==16);
    assert(c.prompt_limit(16, 16)==16);
    c.prefill_budget_ms=1000;
    c.prefill_quantum=128;
    assert(c.prompt_limit(2048, 16)==400); // floor(500/128)*128 + 16
    c.observe_depth(4, 9, 1, 100); // malformed samples must be ignored

}
