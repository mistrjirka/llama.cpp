// Compare complete, teacher-forced output distributions, not only generated strings.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(int argc, char ** argv) {
    try {
        if (argc != 3) { throw std::runtime_error("Usage: compare-logits baseline.bin candidate.bin"); }
        std::ifstream a(argv[1],std::ios::binary), b(argv[2],std::ios::binary);
        uint32_t ar=0,av=0,br=0,bv=0;
        a.read((char*)&ar,4);a.read((char*)&av,4);b.read((char*)&br,4);b.read((char*)&bv,4);
        if (!a || !b || ar!=br || av!=bv || av==0 || av>1000000) {throw std::runtime_error("Invalid output headers");}
        std::vector<float> x(av),y(av);uint32_t changed=0;double error=0,energy=0,max_abs=0,max_prob=0,js_sum=0;
        for(uint32_t row=0;row<ar;++row) {
            a.read((char*)x.data(),av*4);b.read((char*)y.data(),av*4);
            if(!a||!b) {throw std::runtime_error("Truncated output");}
            const auto ix=std::max_element(x.begin(),x.end()),iy=std::max_element(y.begin(),y.end());
            changed+=(ix-x.begin()!=iy-y.begin());double zx=0,zy=0;
            for(uint32_t i=0;i<av;++i) {
                if(!std::isfinite(x[i])||!std::isfinite(y[i])) {throw std::runtime_error("Non-finite logits");}
                const double diff=double(x[i])-y[i];error+=diff*diff;energy+=double(x[i])*x[i];max_abs=std::max(max_abs,std::abs(diff));
                zx+=std::exp(double(x[i]-*ix));zy+=std::exp(double(y[i]-*iy));
            }
            double js=0;
            for(uint32_t i=0;i<av;++i) {
                const double px=std::exp(double(x[i]-*ix))/zx,py=std::exp(double(y[i]-*iy))/zy,m=(px+py)/2;
                max_prob=std::max(max_prob,std::abs(px-py));
                if(px>0)js+=.5*px*std::log(px/m);if(py>0)js+=.5*py*std::log(py/m);
            }
            js_sum+=js;
        }
        std::cout.precision(12);
        std::cout<<"{\"rows\":"<<ar<<",\"vocab\":"<<av<<",\"greedy_differences\":"<<changed
                 <<",\"normalized_mse\":"<<error/std::max(energy,1e-300)<<",\"max_logit_delta\":"<<max_abs
                 <<",\"max_probability_delta\":"<<max_prob<<",\"mean_js_divergence\":"<<js_sum/ar<<"}\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
