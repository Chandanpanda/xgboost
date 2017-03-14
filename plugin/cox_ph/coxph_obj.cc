/*!
 * Copyright 2015 by Contributors
 * \file custom_metric.cc
 * \brief This is an example to define plugin of xgboost.
 *  This plugin defines the additional metric function.
 */
#include <xgboost/base.h>
#include <dmlc/parameter.h>
#include <xgboost/objective.h>

#include <algorithm>
#include <cstdio>
#include <limits>

namespace xgboost {
namespace obj {
// This is a helpful data structure to define parameters
// You do not have to use it.
// see http://dmlc-core.readthedocs.org/en/latest/parameter.html
// for introduction of this module.
struct CoxPHParam : public dmlc::Parameter<CoxPHParam> {
    std::string method;
    // declare parameters
    DMLC_DECLARE_PARAMETER(CoxPHParam) {
        DMLC_DECLARE_FIELD(method).set_default("breslow")
        .describe("Proportional hazard method");
    }
};

DMLC_REGISTER_PARAMETER(CoxPHParam);

// Define a customized logistic regression objective in C++.
// Implement the interface.
class CoxPH : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<bst_float> &preds,
                   const MetaInfo &info,
                   int iter,
                   std::vector<bst_gpair> *out_gpair) override {
      // apply exponential to all predictions
      std::vector<bst_float> exp_preds(preds);
      for (size_t i = 0; i < exp_preds.size(); ++i) {
          exp_preds[i] = std::exp(exp_preds[i]);
      }

      std::vector<bst_ulong> time_to_convert(info.labels.begin(), info.labels.end());
      std::vector<bool> conversion_event(info.censor.begin(), info.censor.end());
      auto max_time = *std::max_element(time_to_convert.begin(), time_to_convert.end());

      std::vector<bst_ulong> d_n(max_time+1, 0);
      calc_dn(conversion_event, time_to_convert, max_time, &d_n);

      std::vector<bst_float> second_part_denom(max_time+1, 0.0f);
      calc_second_part_denom(exp_preds, max_time, time_to_convert, &second_part_denom);
     // calc_second_part_denom_optimized(exp_preds, max_time, time_to_convert, &second_part_denom);
 
      std::vector<bst_float> grad_second_part(max_time+1, 0.0f);
      std::vector<bst_float> hess_part(max_time+1, 0.0f);
      calc_second_part(d_n, second_part_denom, max_time, &grad_second_part, &hess_part);

      // info.label contains conversion_event
      out_gpair->resize(preds.size());
      for (size_t i=0; i<preds.size(); ++i) {
          auto time = time_to_convert[i];
          auto event = conversion_event[i];

          bst_float grad = grad_second_part[time] * exp_preds[i];
          if (event){
                grad = grad - 1.0f;
          }

          bst_float hess = hess_part[time] * exp_preds[i];

          out_gpair->at(i) = bst_gpair(grad, hess);
        }
    }

  void calc_dn(
          const std::vector<bool> &conversion_event,
          const std::vector<bst_ulong> &time_to_convert,
          bst_ulong max_time,
          std::vector<bst_ulong> *out_dn
  ){
      for (size_t i=0;i <time_to_convert.size(); ++i){
              auto event = conversion_event[i];
              auto time = time_to_convert[i];
              if (event){
                  out_dn->at(time)+=1;
              }
          }
  }

  void calc_second_part(
          const std::vector<bst_ulong> &d,
          const std::vector<bst_float> &second_part_denom,
          bst_ulong max_time,
          std::vector<bst_float> *grad_second_part,
          std::vector<bst_float> *hess_part
  )
  {
      for (bst_ulong t=0; t <= max_time ; ++t)
      {
          bst_float grad_second_part_t = 0.0f;
          bst_float grad_second_part_t_2 = 0.0f;
          for (bst_ulong n=0; n<=t; ++n)
          {
              bst_float div1 = d[n] / second_part_denom[n];
              grad_second_part_t+=div1;
              grad_second_part_t_2+= div1 / second_part_denom[n];
          }
          grad_second_part->at(t) = grad_second_part_t;
          hess_part->at(t) = grad_second_part_t - grad_second_part_t_2;
      }
  }

  void calc_second_part_denom(
          const std::vector<bst_float> &exp_preds,
          bst_ulong max_time,
          const std::vector<bst_ulong> &time_to_convert,
          std::vector<bst_float> *out_second_part_denom
  )
  {
     for (size_t t=0; t< out_second_part_denom->size();++t)
      {
          bst_float out_second_part_denom_t = 0.0f;
          for (size_t i=0; i<time_to_convert.size(); ++i){
              if (time_to_convert[i] >= t){
                  out_second_part_denom_t += exp_preds[i];
              }
          }
          out_second_part_denom->at(t) = out_second_part_denom_t;
      }
  }

  void calc_second_part_denom_optimized(
          const std::vector<bst_float> &exp_preds,
          bst_ulong max_time,
          const std::vector<bst_ulong> &time_to_convert,
          std::vector<bst_float> *out_second_part_denom
  )
  {
      // only works if input exp_preds is sorted

      auto prev_time = max_time;
      auto sum_y = 0.0f;
      for (int i = time_to_convert.size()-1; i >=0; --i)
      {
          auto time = time_to_convert[i];
          if (prev_time != time)
          {
              for (int t=prev_time; t>time; --t) {
                  (*out_second_part_denom)[t] = sum_y;
              }
          }
          sum_y+= exp_preds[i];
          prev_time = time;
      }
      for (int t=prev_time; t>=0; --t)
      {
          (*out_second_part_denom)[t] = sum_y;
      }
  }

  const char* DefaultEvalMetric() const override {
    return "partial_likelihood";
  }

 private:
  CoxPHParam param_;
};



// Finally register the objective function.
// After it succeeds you can try use xgboost with objective=mylogistic
XGBOOST_REGISTER_OBJECTIVE(CoxPH, "coxph")
.describe("User define cox proportional hazard regression plugin")
.set_body([]() { return new CoxPH(); });

}  // namespace obj
}  // namespace xgboost
