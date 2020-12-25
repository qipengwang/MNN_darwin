//
//  SGD.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SGD.hpp"
#include "OpGrad.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "Utils.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {
SGD::SGD(std::shared_ptr<Module> module) : ParameterOptimizer(module) {
    auto train = ParameterOptimizer::trainable();
    for (auto p : train) {
        mHistory[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

void SGD::setLearningRate(float rate) {
    mLearningRate = rate;
}

void SGD::setMomentum(float momentum) {
    mMomentum = momentum;
}

void SGD::setWeightDecay(float decay) {
    mWeightDecay = decay;
}

void SGD::setRegularizationMethod(RegularizationMethod method) {
    mRegularizationMethod = method;
}

float SGD::currentLearningRate() {
    return mLearningRate;
}

float SGD::getMomentum() {
    return mMomentum;
}

float SGD::getWeightDecay() {
    return mWeightDecay;
}

SGD::RegularizationMethod SGD::getRegularizationMethod() {
    return mRegularizationMethod;
}

Express::VARP SGD::regularizeParameters(Express::VARP param, Express::VARP grad) {
    VARP addWeightDecayGrad;
    if (mRegularizationMethod == L1) {
        auto temp          = _Sign(param);
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * temp + grad;
    } else if (mRegularizationMethod == L2) {
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * param + grad;
    } else if (mRegularizationMethod == L1L2) {
        auto temp          = _Sign(param);
        auto L1 = _Const(mWeightDecay, {}, NCHW) * temp;
        auto L2 = _Const(mWeightDecay, {}, NCHW) * param;
        addWeightDecayGrad = L1 + L2 + grad;
    }

    return addWeightDecayGrad;
}

Express::VARP SGD::onComputeUpdateValue(Express::VARP param, Express::VARP grad) {
    auto lr         = _Const(mLearningRate, {}, NCHW);
    mHistory[param] = lr * grad + _Const(mMomentum, {}, NCHW) * mHistory[param];
    mHistory[param].fix(Express::VARP::CONSTANT);
    //FUNC_PRINT_ALL(_ReduceMax(grad)->readMap<float>()[0], f);
    return mHistory[param];
}

std::map<Express::VARP, Express::VARP> SGD::onGetNextParameter(Express::VARP loss) {
    MNN_PRINT("mGradBlockExprName = <%s>\n", mGradBlockExprName.c_str());
    printf("num layers = %d\n", module()->nLayers());
    std::map<Express::VARP, Express::VARP> fuck;
    auto parameters = module()->parameters();
    std::vector<VARP> trainable_params;
    for (auto p : parameters) {
        if (nullptr == p.get() || p->expr().first->get() != nullptr) {
            // untrainable
            continue;
        }
        if (p->expr().first->inputType() == Express::VARP::TRAINABLE) {
            trainable_params.emplace_back(p);
        }
    }
    printf("finish get trainable params\n");
    std::map<VARP, VARP> swapable_feature;
    for (auto p: swapable()) {
        auto wpv = p->getInput2Expr();
//        printf("wpv.size() = %lu\n", wpv.size());
        assert(wpv.size() == 1);
//        printf("wpv[0].lock()->outputVars().size() = %lu\n", wpv[0].lock()->outputVars().size());
        assert(wpv[0].lock()->outputVars().size() == 1);
        swapable_feature.insert(std::make_pair(p, VARP(wpv[0].lock()->outputVars()[0].lock())));
    }
    printf("finish get <trainable-params, feature-map>\n");
    std::vector<VARP> prepareCompute;
    std::vector<std::string> tags;
    for (auto iter : parameters) {
        if (iter->expr().first->get() != nullptr) {
            //untrainable掉进来
//            iter->setName(std::to_string(prepareCompute.size()));
            tags.emplace_back("untrainable");
//            if (iter->name().empty()) {
//                iter->setName(&"untrainable:" [ prepareCompute.size()]);
//            }
            prepareCompute.emplace_back(iter);
//            printf("untrainable.exe-order.size = %lu\n", Variable::getExecuteOrder({iter}).size());
        } else if (swapable().find(iter) != swapable().end()) {
            tags.emplace_back("featuremap");
//            if (trainable_feature[iter]->name().empty()) {
//                iter->setName(&"featuremap:" [prepareCompute.size()]);
//            }
            prepareCompute.emplace_back(swapable_feature[iter]);
            assert(swapable_feature[iter]->expr().first->get() != nullptr);
        }
    }
    printf("without unswapable vars, prepareCompute.size = %lu\n", prepareCompute.size());

    auto eo = Variable::getExecuteOrder({loss});
    auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);
    std::map<VARP, VARP> invertedGrad;
    for(auto iter: grad) {
        invertedGrad.insert(std::make_pair(iter.second, iter.first));
//        printf("gradient.input.size = %lu\n", iter.second->expr().first->inputs().size());
    }

    for (auto iter = trainable_params.rbegin(); iter != trainable_params.rend(); iter++) {
        tags.emplace_back("gradient");
//        if (grad[*iter]->name().empty()) {
//            grad[*iter]->setName(&"gradient:" [ prepareCompute.size()]);
//        }
        prepareCompute.emplace_back(grad[*iter]);
    }
    printf("finish get prepareCompute & start apply prepareCompute function\n");

    // assert statements below are ok!
//    assert(fuck.size()==shit.size());
//    for(int i=0; i<fuck.size(); i++) {
////        printf("fuck[%d] == shit[%d] = %d\n", i, i, fuck[i]==shit[i]);
//        assert(fuck[i]==shit[i]);
//    }

//    for(auto expr: executor_order) {
//        printf("expr.inputsize = %lu\n", expr->inputs().size());
//    }
//    for(auto g: grad) {
//        printf("grad.mfrom.input.size = %lu\n", g.second->expr().first->inputs().size());
//    }
//    for(int i=0; i<executor_order.size(); i++){
//        assert(executor_order[i]->outputSize() == 1);
//        executor_order[i]->setName(std::to_string(i));
//        executor_order[i]->setOutputName(0, executor_order[i]->name() + "_" + std::to_string(0));
//    }
//    for(auto & expr : executor_order){
//        auto inputs = expr->inputs();
//        printf("%lu:\t", inputs.size());
//        for(const auto& in: inputs) {
//            assert(!in->name().empty());
//            printf("%s ", in->name().c_str());
//            if(trainable().find(in) != trainable().end()) {
//                printf("\tfind varp in trainable\t");
//            }
//        }
//        printf("\n");
//    }
//    VARP last;
//    int max_exe_size = 0;
//    std::map<VARP, int> num_of_var_as_input; // 每个变量作为input的次数
//    std::set<EXPRP> all_exprp;
//    for (auto& iter : grad) {
//        auto tmp=Variable::getExecuteOrder({iter.second});
//        for(const auto& t: tmp) {
//            all_exprp.insert(t);
//        }
//        if(tmp.size() > max_exe_size) {
//            max_exe_size = tmp.size();
//            last = iter.second;
//        }
////        iter.second->setName(std::to_string(prepareCompute.size()));
//        prepareCompute.emplace_back(iter.second);
//    }
//    auto total_exe_od = Variable::getExecuteOrder({last});
//    printf("all_exprp.size = %lu, total_exe_od.size = %lu\n", all_exprp.size(), total_exe_od.size());
//    std::set<EXPRP> trainable_input_to;
//    std::map<VARP, EXPRP> trainable_exprp;
//    for(const auto& t: Variable::getExecuteOrder({loss})) {
//        for(auto i: t->inputs()) {
//            if(trainable().find(i)!=trainable().end()) {
//                trainable_exprp.insert(std::make_pair(i, t));
//                trainable_input_to.insert(t);
//            }
//            if(num_of_var_as_input.find(i) != num_of_var_as_input.end()) {
//                num_of_var_as_input[i]++;
//            } else {
//                num_of_var_as_input.insert(std::make_pair(i, 1));
//            }
//        }
//    }
//    printf("trainable-as-input.size = %lu\n", trainable_input_to.size());
//    std::set<VARP> all_param(parameters.begin(), parameters.end());
//
//
//    for (const auto& varp: parameters) {
//        int shape = 1;
//        for (int i: varp->expr().first->inside()->mOutputTensors[0]->shape()) {
//            shape *= i;
//        }
//        if (trainable().find(varp) != trainable().end()) {
//            printf("trainable as input for %d times, and its shape=%d\n", num_of_var_as_input[varp], shape);
//        } else {
//            auto  order = Variable::getExecuteOrder({varp});
//            printf("untrainable as input for %d times, and its shape=%d\n", num_of_var_as_input[varp], shape);
//            printf("untrainable: executeOrderSize = %lu,\tinputsize = %lu,\tsize = %d\n", order.size(), varp->expr().first->inputs().size(), varp->getInfo()->size);
//        }
//
//    }
//    auto total_exe_od = Variable::getExecuteOrder({last});
//    for(auto expr: total_exe_od) {
//        assert(expr->mTo.size() == expr->outputSize());
//    }
//    for(int i=0; i<total_exe_od.size(); i++) {
//        if(total_exe_od[i]->name().empty()) {
//            total_exe_od[i]->setName(std::to_string(i));
//        } else {
//            total_exe_od[i]->setName(total_exe_od[i]->name() + "." +std::to_string(i));
//        }
//        assert(total_exe_od[i]->outputSize() == 1 && total_exe_od[i]->inside() != nullptr);
//        total_exe_od[i]->setOutputName(0, total_exe_od[i]->name() + "%0");
//        int shape = 1;
//        for(auto s: total_exe_od[i]->inside()->mOutputTensors[0]->shape()){
//            shape*=s;
//        }
//        printf("%d\t", shape);
//    }
    /*printf("\n");
    for(const auto& ta: grad) {
//        int shape = 1;
//        for(auto s: ta->expr().first->inside()->mOutputTensors[0]->shape()){
//            shape*=s;
//        }
        printf("\t(shape=%d, exe_order_size=%lu)\n", -1, Variable::getExecuteOrder({ta.second}).size());
    }
    printf("\n");
//    for(auto & i : total_exe_od) {
//        for(const auto & j : i->inputs()) {
//            assert(!j->name().empty());
//        }
//    }
    int empty_varname = 0;
    for(auto i: prepareCompute) {
        if(i->name().empty()) {
            printf("<%s: %d>\n", i->expr().first->name().c_str(), i->expr().first->outputSize());
        }
    }
    printf("max_size = %d, empty_varname = %d\n", max_exe_size, empty_varname);
    printf("grad.size = %lu\texecutor_order.size = %lu\tparams.size = %lu\ttrainable.size = %lu\tprepareCompute.size = %lu\n",
           grad.size(), executor_order.size(), parameters.size(), trainable().size(), prepareCompute.size());*/
    //mbnv2: grad.size = 158    executor_order.size = 963       params.size = 262       trainable.size = 158    prepareCompute.size = 262
    Variable::prepareCompute(prepareCompute); // exec.makeCache()
    printf("finish apply prepareCompute function\n");
//    <untrainable, trainable-grad> //104,2     54,1
    std::vector<VARP> replaceOp(prepareCompute.size());
    for (int i = 0; i < prepareCompute.size(); ++i) {
//        printf("current is %s:\t", tags[i].c_str());
        auto info = prepareCompute[i]->getInfo();
        const void *ptr;
        if (swapable().find(prepareCompute[i]) != swapable().end()) {
            ptr = prepareCompute[i]->readMap<void>(true); // 应该是在这里分配了内存
        } else {
            ptr = prepareCompute[i]->readMap<void>(); // 应该是在这里分配了内存
        }
        // readMap 里面有 executor->compute(cache)
        if (nullptr == ptr) {
            MNN_ERROR("Compute error in SGD\n");
            return {};
        }
//        auto newVar = _Const(ptr, info->dim, info->order, info->type);
//        replaceOp[i]= newVar;
        replaceOp[i] = _Const(ptr, info->dim, info->order, info->type);

//        auto exec_ord = Variable::getExecuteOrder({prepareCompute[i]});
//        printf("finish get exec order for preprare-compute[%d] and its size = %lu\n", i, exec_ord.size());
//        int s=0;
//        for(auto e: exec_ord) {
//            if(!e->visited()){
////                printf("e->outputVars().size = %lu\n", e->outputVars().size());
//                s+=e->outputVars()[0].lock()->getInfo()->size;
//                e->setVisited(true);
//            }
//        }
//
//        printf("save into file: <%s>\n", prepareCompute[i]->name().c_str());
//        printf("var[%d] is %s, with size = %d, and the exec-ord.size belong to it is %d\n",
//               i, tags[i].c_str(), prepareCompute[i]->getInfo()->size, s);
        /*if (tags[i] == "untrainable" || tags[i] == "featuremap") {
//            printf("don't need release && continue\n");
        } else  {
            //tags[i]=="gradient"
//            printf("fall in to grad block:\t");
            bool flag=false;
//            printf("gradient.inputSize = %lu\n", prepareCompute[i]->expr().first->inputs().size());
//            Variable::save({prepareCompute[i]}, ("swap/" + prepareCompute[i]->name()).c_str(), true);
            auto var = invertedGrad[prepareCompute[i]];
//            printf("get trainable varp:\t");
            auto miter = std::find(trainable_params.begin(), trainable_params.end(), var) + 1;
//            printf("get next trainable varp\t");
            if (miter != trainable_params.end() && swapable().find(*miter) != swapable().end()){
//                printf("try releasing memory...\t");
                auto nextVarp = *miter;
                assert(nextVarp.get() != nullptr);
                auto nextVarpGrad = grad[*miter];
                assert(nextVarpGrad.get() != nullptr);
                auto gradExecOrd = Variable::getExecuteOrder({nextVarpGrad});
                auto fmExecOrd = Variable::getExecuteOrder({swapable_feature[nextVarp]});
//                printf("<mdzz>\t");
                for(auto it = gradExecOrd.rbegin(); it != gradExecOrd.rend(); it++) {
//                    printf("<shit>\t");
                    if(std::find(fmExecOrd.begin(), fmExecOrd.end(), *it) == fmExecOrd.end()) {
//                        printf("<fuck>\t");
                        bool ok=true;
                        for(const auto& param: parameters) {
                            if(param->expr().first == *it){
                                ok=false;
                                break;
                            }
                        }
                        if(ok) {
//                            printf("releasing...\t");
//                            flag = flag || Utils::releaseMemoryForHostTensor((*it)->inside()->mOutputTensors[0]);
                        }
                    }
                }
            }
            if(flag) {
                printf("finishing releasing Tensor host memory\t");
            } else {
                printf("don't release any memory\t");
            }
            printf("$\n");
        }*/
//        printf("\n");

    }
    printf("finish readmap & start replace-ing\n");
    for (int i = 0; i < prepareCompute.size(); ++i) {
        Variable::replace(prepareCompute[i], replaceOp[i]);
    }
    printf("finish replace & start apply grad to params\n");
    for (auto &iter : grad) {
        // apply regularization
        auto addWeightDecayGrad = regularizeParameters(iter.first, iter.second);
        addWeightDecayGrad.fix(Express::VARP::CONSTANT);
        // apply momentum, etc.
        auto updateValue = this->onComputeUpdateValue(iter.first, addWeightDecayGrad);
        // apply update
        auto newParameter = iter.first - updateValue;
        iter.second = newParameter;
    }
    printf("finish the function %s\n", __FUNCTION__);
    return grad;
}

} // namespace Train
} // namespace MNN
