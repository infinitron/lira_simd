#ifndef LIRA_H
#define LIRA_H

#include <R.h>
#include <Rmath.h>
#include <hwy/base.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <exception>
#include <execution>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

#include "hwy/aligned_allocator.h"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include <cmath>
#include <stdlib.h>
typedef std::stringstream sstr;

namespace hwy {
namespace HWY_NAMESPACE {

typedef AlignedFreeUniquePtr<float[]> uPtr_F;
typedef AlignedFreeUniquePtr<float*[]> uPtr_Fv; //the "view" array
typedef Vec<ScalableTag<float>> vecF;
typedef ScalableTag<float> tagF;

typedef AlignedFreeUniquePtr<size_t[]> uPtr_Int;
const auto& execParUnseq = std::execution::par_unseq;
struct llikeType
{
    double cur; /* the current log likelihood of the iteration */
    double pre; /* the previous log likelihood of the iteration */
};

struct scalemodelType
{
    double scale;     /* the scale parameter */
    double scale_pr;  /* the prior on the total cnt in exposure = ttlcnt_exp */
    double scale_exp; /* the prior exposure in units of the actual exposure */
};

class TempDS
{
  public:
    inline static std::map<std::string, uPtr_F> m_Fltv;
};

class Ops
{
  public:
    //all the operations assume that the length input vectors are a factor of Lanes(d). May fault if not.
    template<typename ArithOp, class MaxTag>
    inline static void v_op(ArithOp t_op, size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_result)
    {
        const MaxTag a, b;
        auto max_lanes = Constants::get_max_lanes();
        size_t i = 0;
        for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
            Store(t_op(Load(a, t_a.get() + i), Load(b, t_b.get() + i)), a, t_result.get() + i);
        }
        if (i < t_npixels) {
            Store(IfThenElseZero(
                    FirstN(d, t_npixels - i), t_op(Load(a, t_a.get() + i), Load(b, t_b.get() + i))),
                  a,
                  t_result.get() + i);
        }
    }

    template<typename MaxTag, class Type>
    inline static void v_div(size_t t_npixels, const uPtr_F& t_a, Type& t_val, uPtr_F& t_result)
    {
        const MaxTag a, b;
        auto max_lanes = Constants::get_max_lanes();
        const auto div_vec = Set(a, t_val);
        size_t i = 0;
        for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
            Store(Div(Load(a, t_a.get() + i), div_vec), a, t_result.get() + i);
        }
        if (i < t_npixels) {
            Store(IfThenElseZero(
                    FirstN(d, t_npixels - i), Div(Load(a, t_a.get() + i), div_vec)),
                  a,
                  t_result.get() + i);
        }
    }

    template<class VecType, class TagType>
    inline static void mul_add_pixels(const size_t& t_npixels, const size_t& max_lanes, const uPtr_F& t_a, const uPtr_F& t_b, VecType& t_result, size_t t_start_a = 0, size_t t_start_b = 0)
    {
        //const ScalableTag<float> a;
        //const ScalableTag<float> b;
        const TagType a, b;

        size_t i = 0;
        for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
            t_result += MulAdd(Load(a, t_a.get() + i + t_start_a), Load(b, t_b.get() + i + t_start_b), t_result);
        }
        if (i < t_npixels) {
            t_result += IfThenElseZero(
              FirstN(a, t_npixels - i), MulAdd(Load(a, t_a.get() + i + t_start_a), Load(b, t_b.get() + i + t_start_b)));
        }
    }

    template<class VecType, class TagType>
    inline static void mul_store_add_pixels(const size_t& t_npixels, const size_t& max_lanes, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_store, VecType& t_result, const size_t& t_start_a = 0, const size_t& t_start_b = 0, const size_t& t_start_store = 0)
    {
        //const ScalableTag<float> a;
        //const ScalableTag<float> b;
        const TagType a, b;
        VecType mul_value;

        size_t i = 0;
        for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
            mul_value = Mul(Load(a, t_a.get() + i + t_start_a), Load(b, t_b.get() + i + t_start_b));
            Store(mul_value, a, t_store.get() + i + t_start_store);
            t_result += mul_value;
        }
        if (i < t_npixels) {
            mul_value = IfThenElseZero(
              FirstN(d, t_npixels - i), Mul(Load(a, t_a.get() + i + t_start_a), Load(b, t_b.get() + i + t_start_b)));
            Store(mul_value, a, t_store.get() + i + t_start_store);
            t_result += mul_value;
        }
    }

    template<class TagType, class T>
    inline static T reduce(const size_t& t_npixels, const uPtr_F& t_map)
    {
        const TagType a, d;
        auto const sum_vec = Zero(TagType);
        auto const max_lanes = Constants::get_max_lanes();
        size_t i = 0;

        //__PAR__
        for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
            sum_vec = Add(Load(a, t_map.get() + i), sum_vec);
        }

        if (i < t_npixels) {
            sum_vec = IfThenElseZero(FirstN(d, t_npixels - i), Load(a, t_map.get() + i)) + sum_vec;
        }

        return static_cast<T>(GetLanes(SumOfLanes(sum_vec)));
    }
    inline static void redistribute_counts(PSF& t_psf, CountsMap& t_deblur, CountsMap& t_obs, llikeType& t_llike)
    {
        t_deblur.set_data_zero();

        const auto max_lanes = Constants::get_max_lanes();
        const auto psf_dim = t_psf.get_dim();
        const auto psf_npixels = t_psf.get_npixels();
        const auto wmap_dim = t_deblur.get_wmap_dim();
        const auto deblur_pad_dim = t_deblur.get_pad_dim();
        const auto deblur_dim = t_deblur.get_dim();
        const auto obs_npixels = t_obs.get_npixels();
        const uPtr_F& deblur_wmap = t_deblur.get_warped_img();
        const uPtr_F& psf_rmat = t_psf.get_rmat();
        const uPtr_F& obs_map = t_obs.get_data_map();
        uPtr_Fv& deblur_wmap_data_ref = t_deblur.get_wmap_data_ref();
        uPtr_F& psf_inv = t_psf.get_inv();
        tagF d;

        //loop over each pixel in the observation
        //This equivalent operation is window-multiplying the warped obs_matrix with the PSF matrix starting from the right-top edge
        size_t deblur_start_idx = 0;
        size_t psf_start_idx = 0;
        size_t start_idx = 0;
        vecF sum_vec;
        float count = 0;
        float sum;
        float p_total = 1, p_cur;
        float sample;

        for (size_t row = 0; row < deblur_dim; ++row) {
            for (size_t col = 0; col < deblur_dim; ++col) {
                deblur_start_idx = row * deblur_dim + col;
                sum_vec = Zero(d);
                count = obs_map.get()[deblur_start_idx];
                sum = 0;

                //multiply with warped image with the PSF centered at the current pixel
                for (size_t i = 0; i < psf_dim; ++i) {
                    Ops::mul_store_add_pixels<vecF, tagF>(psf_dim, max_lanes, deblur_wmap, psf_rmat, psf_inv, sum_vec, ((row + i) * deblur_dim + col), i * psf_dim, i * psf_dim);
                }

                //compute the sum and normalize the inverse PSF
                sum = GetLane(SumOfLanes(sum_vec));
                t_psf.normalize_inv(sum);

                //update the log likelihood
                if (count > 0)
                    t_llike.cur += obs_map.get()[deblur_start_idx] * log(sum);
                t_llike.cur -= sum;

                if (sum == 0.0 && obs_map.get()[deblur_start_idx] > 0) {
                    throw(InconsistentData(row, col)); //psf doesn't allow data in this pixel
                }

                //Multinomial calculations
                if (count > 0) {
                    //__PAR__
                    for (size_t i = 0; i < psf_npixels; ++i) {
                        if (psf_inv.get()[i] > 0) {
                            p_cur = psf_inv.get()[i] / p_total;
                            sample = (p_cur < 1.f) ? rbinom(count, p_cur) : count;
                        } else
                            sample = 0;

                        p_total -= psf_inv.get()[i];
                        psf_inv.get()[i] = sample;
                    }

                    //redistribute the counts to the deblur data map
                    //do not parallelize this loop as multiple elements in the ref map point to the same location!
                    for (size_t i = 0; i < psf_dim; ++i) {
                        for (size_t j = 0; j < psf_dim; ++j) {
                            *deblur_wmap_data_ref.get()[deblur_start_idx + i * deblur_dim + j] += psf_inv.get()[i * psf_dim + j];
                        }
                    }
                } //multinomial calc loop
            }     //deblur col loop
        }         //deblur row loop
    }

    inline static void remove_bkg_from_data(CountsMap& t_deblur, CountsMap& t_src, CountsMap& t_bkg, const scalemodelType& bkg_scale)
    {
        auto npixels = t_src.get_npixels();
        auto max_lanes = Constants::get_max_lanes();
        tagF a, b, c, d;
        if (TempDS::m_Fltv.count("rem_bkg_prob_src") == 0) {
            TempDS::m_Fltv["rem_bkg_prob_src"] = AllocateAligned<float>(npixels);
        }

        auto& prob_src = TempDS::m_Fltv["rem_bkg_prob_src"];
        auto& src_img = t_src.get_img_map();
        auto& src_data = t_src.get_data_map();
        auto& bkg_data = t_bkg.get_data_map();
        auto& bkg_img = t_bkg.get_img_map();
        auto& deblur_data = t_deblur.get_data_map();
        auto bkg_scale_vec = Set(d, bkg_scale.scale);

        auto& prob_src = TempDS::m_Fltv["rem_bkg_prob_src"];

        //reset the data store
        std::fill(execParUnseq, prob_src.get(), prob_src.get() + npixels, 0.f);

        //compute the probability for src: src/(bkg_scale * bkg+src)
        //__PAR__
        for (size_t i = 0; i < npixels; i += max_lanes) {
            Store(Load(a, src_img.get() + i) / (MulAdd(bkg_scale_vec, Load(b, bkg_img.get() + i), Load(c, src_img.get() + i))), d, prob_src.get() + i);
        }

        //loop over each pixel
        //__PAR__
        for (size_t i = 0; i < npixels; ++i) {
            if (prob_src.get()[i] < 1) {
                src_data.get()[i] = rbinom(deblur_data.get()[i], prob_src.get()[i]);
                bkg_data.get()[i] = deblur_data.get()[i] - src_data.get()[i];
            } else {
                src_data.get()[i] = deblur_data.get()[i];
                bkg_data.get()[i] = 0.f;
            }
        }
    }

    inline static void add_cnts_2_adjust_4_exposure(const ExpMap& t_exp_map, CountsMap& t_counts_map)
    {
        auto npixels = t_counts_map.get_npixels();
        auto max_lanes = Constants::get_max_lanes();
        tagF a, b, c, d;
        if (TempDS::m_Fltv.count("exp_missing_count") == 0) {
            TempDS::m_Fltv["exp_missing_count"] = AllocateAligned<float>(npixels);
        }

        const auto& prod_map = t_exp_map.get_prod_map();
        const auto& counts_img = t_counts_map.get_img_map();
        auto& counts_data = t_counts_map.get_data_map();
        auto& exp_missing_count = TempDS::m_Fltv["exp_missing_count"];

        //reset the temp DS
        std::fill(execParUnseq, exp_missing_count.get(), exp_missing_count.get() + npixels, 0.f);

        //compute expected number of missing count
        //__PAR__
        for (size_t i = 0; i < npixels; i += max_lanes) {
            Store(
              NegMulAdd(Load(a, prod_map.get() + i), Load(b, counts_img.get() + i), Load(c, counts_img.get() + i)), d, exp_missing_count.get());
        }

        //__PAR__
        for (size_t i = 0; i < npixels; ++i) {
            exp_missing_count.get()[i] = rpois(exp_missing_count.get()[i]);
        }

        //add again
        //__PAR__
        for (size_t i = 0; i < npixels; i += max_lanes) {
            Store(Load(a, counts_data.get() + i) + Load(b, exp_missing_count.get() + i), d, counts_data.get() + i);
        }
    }

    inline static void check_monotone_convergence() {}

    inline static float update_image_ms(AsyncFileIO& t_out_files, const ExpMap& t_expmap, CountsMap& t_src, MultiScaleMap& t_ms)
    {
        auto dim = t_src.get_dim();
        auto spin_row = Utils::binary_roulette(dim);
        auto spin_col = Utils::binary_roulette(dim);
        float log_prior = 0.0f;

        //_____FILE_IO_____//

        //copy src data into level 0
        t_src.get_spin_data(t_ms.get_level(0).get_map(), spin_row, spin_col);

        //compute_aggreggations
        t_ms.compute_cascade_agregations();

        //update alpha
        update_alpha_ms(t_out_files, t_ms);

        //compute proportions
        t_ms.compute_cascade_proportions();

        //update the expected total count
        t_ms.set_total_exp_count();

        //_____FILE_IO_____//

        //compute log prior
        log_prior = (t_ms.get_ttlcnt_pr() - 1) * log(t_ms.get_max_agg_value()) - t_ms.get_ttlcnt_exp() * t_ms.get_max_agg_value();

        log_prior += t_ms.get_log_prior_n_levels();

        //compute the log scale image
        t_ms.compute_cascade_log_scale_images();

        //store new image in src.img
        t_src.set_spin_img(t_ms.get_level(0).get_map(),spin_row,spin_col);

        return log_prior;
    }

    inline static void update_alpha_ms(AsyncFileIO& t_out_file, MultiScaleMap& t_ms)
    {
        float lower, //lower
          middle,    //middle
          upper,     //upper
          dl_lower,  //dl
          dl_upper,  //dl
          dl_middle; //dl

        auto nlevels = t_ms.get_nlevels();

        for (auto level = 0; level < nlevels; ++level) {
            //initialize lower and upper
            lower = 1.0f;

            while (Ops::dlpost_lalpha(lower, t_ms, level) < 0)
                lower /= 2.0f;

            dl_lower = Ops::dlpost_lalpha(lower, t_ms, level);

            upper = lower * 2.0f;

            while (Ops::dlpost_lalpha(upper, t_ms, level) < 0)
                upper /= 2.0f;

            dl_upper = Ops::dlpost_lalpha(upper, t_ms, level);

            while ((upper - lower) > Constants::convg_bisect) {
                middle = (lower + upper) / 2.0f;
                dl_middle = Ops::dlpost_lalpha(middle, t_ms, level);

                if (dl_middle > 0) {
                    lower = middle;
                    dl_lower = dl_middle;
                } else {
                    upper = middle;
                    dl_upper = dl_middle;
                }
            } //bisect loop
            t_ms.set_alpha(level, update_alpha_ms_MH((upper + lower) / 2.0f, t_ms, level));

            //_____FILE IO_____//
        }
    }

    inline static float update_alpha_ms_MH(const float& t_prop_mean, MultiScaleMap& t_ms, const size_t& t_level)
    {
        float proposal, //
          current,      //
          prop_sd,      //
          lg_prop_mn,   //
          log_ratio;    //

        current = t_ms.get_alpha(t_level);
        prop_sd = ddlpost_lalpha(t_prop_mean, t_ms, t_level);
        if (t_prop_mean * sqrt(prop_sd) < 1e-10) {
            throw(InconsistentData("Infinite MH proposal variance in update_alpha_ms_MH"));
        } else
            prop_sd = Constants::MH_sd_inflate / sqrt(prop_sd);

        for (auto i = 0; i < Constants::MH_iter; ++i) {
            proposal = rlnorm(lg_prop_mn, prop_sd); /* log normal proposal dist'n */
            proposal = rlnorm(lg_prop_mn, prop_sd); /* log normal proposal dist'n */
            log_ratio =
              lpost_lalpha(proposal, t_ms, t_level) - lpost_lalpha(current, t_ms, t_level) + dlnorm(current, lg_prop_mn, prop_sd, 1) - dlnorm(proposal, lg_prop_mn, prop_sd, 1);
            if (runif(0, 1) < exp(log_ratio))
                current = proposal;
        }
        return (current);
    };

    inline static float ddlpost_lalpha(float t_alpha, MultiScaleMap& t_ms, size_t level)
    {
    }

    template<typename D, int Pow4>
    inline static float dnlpost_alpha(D t_dfunc, float& t_alpha, MultiScaleMap& t_ms, const size_t& t_level)
    {
        auto factor = float(pow(4, Pow4));
        //D=digamma=>dlpost_alpha
        //D=trigamma=>ddlpost_alpha

        auto dim = t_ms.get_level(t_level).get_dim();
        float dlogpost = dim * dim * (factor * t_dfunc(4.0 * t_alpha) - 4.0 * t_dfunc(t_alpha));

        dlogpost += t_ms.get_level(t_level).get_ngamma_sum<D>(D(), t_alpha) - factor * t_ms.get_level(t_level + 1).get_ngamma_sum<D>(D(), 4 * t_alpha) + t_ms.get_log_prior(t_alpha, Pow4);
    }

    inline static float dlpost_lalpha(float t_alpha, MultiScaleMap& t_ms, const size_t& t_level)
    {
        return dnlpost_alpha<f_digamma, 1>(f_digamma(), t_alpha, t_ms, t_level) * t_alpha + 1.f;
    }

    inline static float ddlpost_lalpha(float& t_alpha, MultiScaleMap& t_ms, const size_t& t_level)
    {
        return dnlpost_alpha<f_trigamma, 1>(f_trigamma(), t_alpha, t_ms, t_level) * t_alpha + dnlpost_alpha<f_trigamma, 2>(f_trigamma(), t_alpha, t_ms, t_level) * t_alpha * t_alpha;
    }

    inline static float lpost_lalpha(float& t_alpha, MultiScaleMap& t_ms, const size_t& t_level)
    {
        return dnlpost_alpha<f_lgamma, 0>(f_lgamma(), t_alpha, t_ms, t_level) + log(t_alpha);
    }
};

class AsyncFileIO
{
  public:
    AsyncFileIO() {}
};

class Utils
{
  public:
    inline static size_t get_power_2(size_t i)
    {
        if ((i & (i - 1)) == 0) {
            return Constants::MultiplyDeBruijnBitPosition2[(uint32_t)(i * 0x077CB531U) >> 27];
        }
    }

    inline static size_t binary_roulette(size_t dim)
    {
        return (size_t)dim * runif(0, 1);
    }
};

struct f_digamma
{
    float operator()(float& v) { return digamma(v); }
};

struct f_trigamma
{
    float operator()(float& v) { return trigamma(v); }
};

struct f_lgamma
{
    float operator()(float& v) { return lgammafn(v); }
};

struct f_vadd{
    vecF operator()(vecF &t_a,vecF &t_b){return Add(t_a,t_b);}
};

class Constants
{
  public:
    // clang-format off
    inline static const int MultiplyDeBruijnBitPosition2[32]
    {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
        8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
    };
;
    // clang-format on
    inline static const std::map<MapType, std::string> map_names = { { MapType::PSF, "PSF" }, { MapType::EXP, "Exposure" }, { MapType::COUNTS, "Counts" } };

    inline static const float convg_bisect{ 1e-8 };
    // inline static const size_t get_lanes()
    // {
    //     ScalableTag<float> d;
    //     return static_cast<size_t>(Lanes(d));
    // }
    inline static const ScalableTag<float> d;
    typedef decltype(d) maxVec;
    inline static const size_t get_max_lanes()
    {
        return Lanes(d);
    };
    inline static const float MH_sd_inflate{ 2.0 };
    inline static const int MH_iter{ 10 };
    inline static const int NR_END{ 1 };
    inline static const int MAX_ITER_WHILE{ 100 };

  protected:
};

enum class MapType
{
    PSF,
    EXP,
    COUNTS
};

enum class CountsMapDataType
{
    DATA,
    IMG
};

enum class DFunc
{
    DIGAMMA,
    TRIGAMMA
};

class IncorrectDimensions : virtual public std::exception
{
  protected:
    std::stringstream m_err_stream;

  public:
    explicit IncorrectDimensions(const std::string& t_msg)
      : m_err_stream(t_msg){};

    explicit IncorrectDimensions(size_t t_x, size_t t_y, std::string t_map_name, bool t_square_err = FALSE)
    {
        if (t_square_err) {
            m_err_stream << 'Incorrect ' << t_map_name << ' map dimensions: ' << t_map_name << ' map must be a square.\n';
            m_err_stream << ' Input dimensions nrows: ' << t_x << ', ncols: ' << t_y;
        } else {
            m_err_stream << "Incorrect " << t_map_name << " map dimensions: Each side must be a power of 2 and greater than 8.\n";
            m_err_stream << "Input dimensions: nrows: " << t_x << " ncols: " << t_y;
        }
    }

    virtual const char* what() const throw()
    {
        return m_err_stream.str().c_str();
    }
};

class IncompleteInitialization : virtual public std::exception
{
  protected:
    std::stringstream m_err_stream;

  public:
    explicit IncompleteInitialization(const std::string& t_msg)
      : m_err_stream(t_msg){};

    explicit IncompleteInitialization(const std::string& t_map_name, const std::string& t_msg)
    {
        m_err_stream << "Incomplete initialization in the map " << t_map_name << '\n'
                     << t_msg;
    }

    virtual const char* what() const throw()
    {
        return m_err_stream.str().c_str();
    }
};

class InconsistentData : virtual public std::exception
{
  protected:
    std::stringstream m_err_stream;

  public:
    explicit InconsistentData(const std::string& t_msg)
    {
        m_err_stream << "Inconsistent data error: " << t_msg;
    };

    explicit InconsistentData(const size_t& t_nrow, const size_t& t_ncol)
    {
        m_err_stream << "Inconsistent data error: The PSF does not allow data in pixel row(" << t_nrow << ") col(" << t_ncol << ")\n";
    };

    virtual const char* what() const throw()
    {
        return m_err_stream.str().c_str();
    }
};

class InvalidParams : virtual public std::exception
{
  protected:
    std::stringstream m_err_stream;

  public:
    explicit InvalidParams(const std::string& t_msg)
    {
        m_err_stream << "Invalid parameters: " << t_msg;
    };

    explicit InvalidParams(const size_t& spin_row, const size_t& spin_col)
    {
        m_err_stream << "Invalid params: The spins can only be 0 or 1. spin_row(" << spin_row << ") spin_col(" << spin_col << ")\n";
    }

    virtual const char* what() const throw()
    {
        return m_err_stream.str().c_str();
    }
};

template<typename T>
class ImgMap
{
  public:
    ImgMap(size_t t_nrows, size_t t_ncols, MapType t_map_type, std::string t_name)
      : m_nrows(t_nrows)
      , m_ncols(t_ncols)
      , m_map_type(t_map_type)
      , m_map_name(t_name)
      , m_npixels(m_nrows * m_ncols)
    {
        check_sizes();
        static_cast<T*>(this).initialize();

        if (m_map_type != MapType::PSF) {
            //http://www.graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
            //fancy way to compute the power of two
            m_max_levels = Utils::get_power_2(m_nrows);
        }
    }

    size_t get_npixels() const
    {
        return m_npixels;
    }

    const size_t get_dim() const
    {
        return m_nrows;
    }

  protected:
    const size_t m_nrows;
    const size_t m_ncols;
    size_t m_max_levels{ 1 };
    MapType m_map_type;
    std::string m_map_name;
    size_t m_npixels;

  private:
    void check_sizes()
    {
        if (m_nrows != m_ncols) {
            throw(IncorrectDimensions(m_nrows, m_ncols, m_map_name, true));
        }

        if (m_map_type == MapType::PSF && m_nrows % 2 == 0) {
            std::cout << 'Warning: The PSF must have an odd dimension to allow the maximum to be exactly at the center of the image';
        }

        if (m_map_type != MapType::PSF && (m_nrows == 0 || (m_nrows & (m_nrows - 1)) || m_nrows < 8)) {
            throw(IncorrectDimensions(m_nrows, m_ncols, m_map_name, FALSE));
        }
    }
};

class WrappedIndexMap
{
  public:
    WrappedIndexMap(size_t& t_img_dim, size_t& t_psf_dim)
      : m_img_dim(t_img_dim)
      , m_psf_dim(t_psf_dim)
    {
        m_pad_size = m_psf_dim / 2; //assumes an odd sized PSF
        m_pad_img_dim = m_pad_size + m_img_dim;
        m_npixels = pow(2, m_pad_img_dim);
        size_t _a, _b;

        for (size_t i = 0; i < m_pad_img_dim; ++i) {
            for (size_t j = 0; j < m_pad_img_dim; ++j) {
                //use this map to create a warped image and simply multiply it with the psf matrix between the edges

                m_idx_map.push_back(wrap_idx(i - m_pad_size, m_pad_img_dim) * m_pad_img_dim + wrap_idx(j - m_pad_size, m_pad_img_dim));
            }
        }
    }

    const std::vector<size_t>& get_map() const
    {
        return m_idx_map;
    }

    const size_t& get_npixels() const
    {
        return m_npixels;
    }

    const size_t& get_dim() const
    {
        return m_pad_img_dim;
    }

    const size_t& get_pad_dim() const
    {
        return m_pad_size;
    }

  protected:
    size_t m_img_dim;
    size_t m_psf_dim;
    size_t m_pad_img_dim;
    size_t m_pad_size;             //Padding on each side of the image. Elements in the padded region will be wrapped
    size_t m_npixels;              //total pixels in the map inclusive of the padding
    std::vector<size_t> m_idx_map; //Each value maps the pixels from the input image to its padded/wrapped counterpart. The total padding is determined by the PSF's size.

  private:
    size_t wrap_idx(size_t t_idx, const size_t& t_dim)
    {
        if (t_idx >= 0)
            return t_idx;

        while (t_idx < 0)
            t_idx += 1;
        return t_idx % t_dim;
    }
};
class PSF : public ImgMap<PSF>
{
  public:
    PSF(size_t t_nrows, size_t t_ncols, const float* t_psf_mat, std::string t_name)
      : ImgMap<PSF>{ t_nrows, t_ncols, MapType::PSF, t_name }
      , m_mat_holder(t_psf_mat)
    {
    }

    PSF& initialize()
    {

        //ScalableTag<float> d;
        //m_aligned_size = m_orig_size + m_orig_size % Lanes(d); //pad the extra elements with zeros

        m_mat = AllocateAligned<float>(m_npixels);
        m_inv = AllocateAligned<float>(m_npixels);

        std::copy(execParUnseq, m_mat_holder, m_mat_holder + m_npixels, m_mat.get());
        //std::copy(execParUnseq, m_mat_holder, m_mat_holder + m_npixels, m_rmat.get());
        std::reverse_copy(execParUnseq, m_mat_holder, m_mat_holder + m_npixels, m_rmat.get());
        std::fill(execParUnseq, m_mat.get() + m_orig_size, m_mat.get() + m_aligned_size, 0.f);

        //TODO--compute L,R,U,D
    }

    const uPtr_F& get_rmat() const
    {
        return m_rmat;
    }

    uPtr_F& get_inv()
    {
        return m_inv;
    }

    void normalize_inv(float& sum)
    {
        Ops::v_div<tagF, float>(m_npixels, m_inv, sum, m_inv);
    }

  protected:
    uPtr_F m_mat;
    uPtr_F m_inv;
    uPtr_F m_rmat; //180 deg CCW rotated matrix, a.k.a reversed
    size_t m_L{ 0 }, m_R{ 0 }, m_U{ 0 }, m_D{ 0 };
    size_t m_orig_size;
    size_t m_aligned_size;

  private:
    const float* m_mat_holder; //temporary storage before initialization checks
};

class ExpMap : public ImgMap<ExpMap>
{
  public:
    ExpMap(size_t t_nrows, size_t t_ncols, const float* t_expmap, std::string t_name)
      : m_mat_holder(t_expmap)
      , ImgMap<ExpMap>(t_nrows, t_ncols, MapType::EXP, t_name)
    {
    }

    void initialize()
    {
        m_map = AllocateAligned<float>(n_pixels);
        //pr_det = AllocateAligned<float>(n_pixels);
        m_prod = AllocateAligned<float>(n_pixels);

        std::copy(m_mat_holder, m_mat_holder + n_pixels, m_map.get());
        max_exp = *std::max_element(execParUnseq, m_map.get(), m_map.get() + n_pixels);

        ScalableTag<float> d;
        const auto max_vector = Set(d, max_exp);

        //normalize the exposure map
        for (size_t i = 0; i < n_pixels; i += Lanes(d)) {
            const auto a = Load(d, m_map.get() + i);
            Store(Div(a, max_vector), d, m_map.get() + i);
        }

        //Because wrapping is the enforced default behaviour => pr_det=1, hence prod=m_map
        std::copy(execParUnseq, m_map.get(), m_map.get() + n_pixels, m_prod.get());
    }

    const uPtr_F& get_prod_map() const
    {
        return m_prod;
    }

  protected:
    uPtr_F m_map;
    uPtr_F pr_det;
    uPtr_F m_prod;

  private:
    const float* m_mat_holder;
    size_t n_pixels{ 0 };
    float max_exp{ 1.f };
    size_t max_levels{ 1 };
};

class CountsMap : public ImgMap<CountsMap>
{
  public:
    CountsMap(size_t t_nrows, size_t t_ncols, float* t_map, CountsMapDataType t_map_data_type, std::string t_name)
      : ImgMap(t_nrows, t_ncols, MapType::COUNTS, t_name)
      , m_map_holder(t_map)
      , map_data_type(t_map_data_type)
    {
    }

    CountsMap& initialize()
    {
        m_data = AllocateAligned<float>(m_npixels);
        m_img = AllocateAligned<float>(m_npixels);

        if (map_data_type == CountsMapDataType::DATA) {
            std::copy(m_map_holder, m_map_holder + m_npixels, m_data.get());
            set_img_zero();
        } else {
            std::copy(m_map_holder, m_map_holder + m_npixels, m_img.get());
            set_data_zero();
        }
    }

    void set_data_zero()
    {
        std::fill(m_data.get(), m_data.get() + m_npixels, 0.f);
    }

    void set_img_zero()
    {
        std::fill(m_img.get(), m_img.get() + m_npixels, 0.f);
    }

    uPtr_F& get_data_map()
    {
        return m_data;
    }

    uPtr_F& get_img_map()
    {
        return m_img;
    }

    /*void set_warped_img(const WrappedIndexMap& t_w_idx_map)
    {
        m_is_wimg_set = true;
        m_pad_dim = t_w_idx_map.get_pad_dim();
        m_npixels_wmap = t_w_idx_map.get_npixels();
        m_wmap_dim = t_w_idx_map.get_dim();
        if (m_warped_img_ref == nullptr) {
            m_warped_img_ref = AllocateAligned<float*>(m_npixels_wmap);
            for (size_t i = 0; i < m_npixels_wmap; ++i)
                m_wmap_seq_idx.push_back(i);
        }
        if (m_warped_img == nullptr) {
            m_warped_img = AllocateAligned<float>(m_npixels_wmap);
        }
        const auto map = t_w_idx_map.get_map();

        for (auto i : m_wmap_seq_idx) {
            *m_warped_img_ref.get()[i] = m_img.get()[i];
        }
        // std::for_each(execParUnseq, m_wmap_seq_idx.begin(), m_wmap_seq_idx.end(), [&](const auto &i) {
        //     *m_warped_img.get()[i] = &map[i];
        // });
        //std::transform(execParUnseq,)
    }*/

    void set_warped_mat(const WrappedIndexMap& t_w_idx_map, CountsMapDataType t_type = CountsMapDataType::DATA)
    {
        m_pad_dim = t_w_idx_map.get_pad_dim();
        m_npixels_wmap = t_w_idx_map.get_npixels();
        m_wmap_dim = t_w_idx_map.get_dim();

        if (m_wmap_seq_idx.size() != m_npixels_wmap) {
            for (size_t i = 0; i < m_npixels_wmap; ++i)
                m_wmap_seq_idx.push_back(i);
        }

        if (t_type == CountsMapDataType::IMG) {
            m_is_wimg_set = true;
            if (m_warped_img_ref == nullptr) {
                m_warped_img_ref = AllocateAligned<float*>(m_npixels_wmap);
            }
            if (m_warped_img == nullptr) {
                m_warped_img = AllocateAligned<float>(m_npixels_wmap);
            }
            for (auto i : m_wmap_seq_idx) {
                m_warped_img_ref.get()[i] = &m_img.get()[i];
            }
        } else {
            m_is_wdata_set = true;
            if (m_warped_data_ref == nullptr) {
                m_warped_data_ref = AllocateAligned<float*>(m_npixels_wmap);
            }
            if (m_warped_data == nullptr) {
                m_warped_data = AllocateAligned<float>(m_npixels_wmap);
            }
            for (auto i : m_wmap_seq_idx) {
                m_warped_data_ref.get()[i] = &m_data.get()[i];
            }
        }
    }

    void init_data_spin_views()
    {
        if (m_is_spin_data_views_set)
            return;

        m_is_spin_data_views_set = true;
        for (int spin_row = 0; spin_row < 2; ++spin_row) {
            for (int spin_col = 0; spin_col < 2; ++spin_col) {
                auto pair_key = std::pair<int, int>(spin_row, spin_col);
                m_spin_data_views[pair_key] = AllocateAligned<float*>(m_npixels);
                auto& spin_view_ref = m_spin_data_views[pair_key];

                //set the view
                for (size_t row = 0; row < m_nrows; ++row) {
                    for (size_t col = 0; col < m_ncols; ++col) {
                        spin_view_ref.get()[row * m_nrows + m_ncols] = &m_data.get()[((row + spin_row) % m_nrows) * m_nrows + (col + spin_col) % m_ncols];
                    }
                }
            }
        }
    }

    void init_img_spin_views(){
        if (m_is_spin_img_views_set)
            return;

        m_is_spin_img_views_set = true;
        for (int spin_row = 0; spin_row < 2; ++spin_row) {
            for (int spin_col = 0; spin_col < 2; ++spin_col) {
                auto pair_key = std::pair<int, int>(spin_row, spin_col);
                m_spin_img_views[pair_key] = AllocateAligned<float*>(m_npixels);
                auto& spin_view_ref = m_spin_img_views[pair_key];

                //set the view
                for (size_t row = 0; row < m_nrows; ++row) {
                    for (size_t col = 0; col < m_ncols; ++col) {
                        spin_view_ref.get()[row * m_nrows + m_ncols] = &m_img.get()[((row + spin_row) % m_nrows) * m_nrows + (col + spin_col) % m_ncols];
                    }
                }
            }
        }
    }



    void get_spin_data(uPtr_F& t_out, size_t spin_row, size_t spin_col)
    {
        auto pair_key = std::pair<int, int>(spin_row, spin_col);
        if (m_spin_data_views.count(pair_key) == 0)
            throw(InvalidParams(spin_row, spin_col));

        const auto& data_spin_ref = m_spin_data_views[pair_key];

        //__PAR__
        for (size_t i = 0; i < m_npixels; ++i) {
            t_out.get()[i] = *data_spin_ref.get()[i];
        }
    }

    void set_spin_img(uPtr_F &t_in_map,size_t spin_row,size_t spin_col){
        auto pair_key = std::pair<int, int>(spin_row, spin_col);
        if (m_spin_img_views.count(pair_key) == 0)
            throw(InvalidParams(spin_row, spin_col));

        const auto& img_spin_ref = m_spin_img_views[pair_key];

        //__PAR__
        for(size_t i=0;i<m_npixels;++i){
            *img_spin_ref.get()[i]=t_in_map.get()[i];
        }
    }

    //returns a const ref to the warped image map. This will be used to redisribute counts (i.e., convolving the map with the PSF)
    const uPtr_F& get_warped_img()
    {
        check_wmap_set();
        for (auto i : m_wmap_seq_idx) {
            m_warped_img.get()[i] = *m_warped_img_ref.get()[i];
        }

        return m_warped_img;
    }

    //returns a reference to the warped data map. Used to update data map in the multinomial calculations
    uPtr_Fv& get_wmap_data_ref()
    {
        return m_warped_data_ref;
    }

    const size_t get_pad_dim() const
    {
        return m_pad_dim;
    }

    size_t get_wmap_dim() const
    {
        check_wmap_set();
        return m_wmap_dim;
    }

  protected:
    uPtr_F m_data;                       //the counts
    uPtr_F m_img;                        //the image (expected counts)
    CountsMapDataType map_data_type;     //type of the data that should be read in
    uPtr_Fv m_warped_img_ref{ nullptr }; //warped (padded) img reference
    uPtr_F m_warped_img{ nullptr };
    uPtr_Fv m_warped_data_ref{ nullptr };
    uPtr_F m_warped_data{ nullptr };
    size_t m_npixels_wmap;
    bool m_is_wimg_set{ false };
    bool m_is_wdata_set{ false };
    size_t m_wmap_dim;
    size_t m_pad_dim{ 0 };
    //std::vector<uPtr_Fv> m_4spin_views;
    std::map<std::pair<int, int>, uPtr_Fv> m_spin_data_views; //holds references to the data map view for different row/col spin values
    std::map<std::pair<int, int>, uPtr_Fv> m_spin_img_views;
    bool m_is_spin_data_views_set{ false };
    bool m_is_spin_img_views_set{ false };

  private:
    const float* m_map_holder;
    std::vector<size_t> m_wmap_seq_idx;
    void check_wmap_set() const
    {
        if (!m_is_wimg_set) {
            throw(IncompleteInitialization(m_map_name, "The warped image is not set yet. Call set_warped_img first."));
        }
    }
};

class MultiScaleLevelMap
{
  public:
    MultiScaleLevelMap(size_t t_dimension, float t_alpha)
      : m_dimension(t_dimension)
      , m_dimension_agg(m_dimension / 2)
      , m_npixels(pow(t_dimension, 2))
      , m_npixels_agg(m_npixels / 4)
      , m_current_map(AllocateAligned<float>(m_npixels))
      , m_temp_storage(AllocateAligned<float>(m_npixels))
      , m_alpha(t_alpha)
    {
        /*, m_row_interleaved_sum(AllocateAligned<float>(m_half_npixels)), m_row_interleaved_A(AllocateAligned<float>(m_half_npixels)), m_row_interleaved_B(AllocateAligned<float>(m_npixels_agg)), m_col_interleaved_A(AllocateAligned<float>(m_npixels_agg)), m_row_to_col_interleaved_flag(m_half_npixels, true), m_row_to_col_indices(m_half_npixels, 0), m_row_interleaved_indices(m_half_npixels, 0), m_sub_npixels(m_npixels / 4), m_half_npixels(m_npixels / 2), m_agg_indices(m_npixels_agg, 0), m_curr_map_agg_indices(m_npixels, 0), m_curr_map_indices(m_npixels, 0),, m_agg_norm_map(AllocateAligned<float>(m_npixels))*/

        std::fill(execParUnseq, m_current_map.get(), m_current_map.get() + m_npixels, 0.f);
        std::fill(execParUnseq, m_temp_storage.get(), m_temp_storage.get() + m_npixels, 0.f);
        //std::fill(m_agg_norm_map.get(), m_agg_norm_map.get() + m_npixels, 0.f);
        /*std::fill(m_row_interleaved_sum.get(), m_row_interleaved_sum.get() + m_half_npixels, 0.f);
                std::fill(m_row_interleaved_A.get(), m_row_interleaved_A.get() + m_half_npixels, 0.f);
                std::fill(m_row_interleaved_B.get(), m_row_interleaved_B.get() + m_half_npixels, 0.f);
                std::fill(m_col_interleaved_A.get(), m_col_interleaved_A.get() + m_npixels_agg, 0.f);
                std::fill(m_col_interleaved_B.get(), m_col_interleaved_B.get() + m_npixels_agg, 0.f);*/

        //pre-compute slicing indices
        // size_t counter = 0;
        // std::for_each(m_row_to_col_interleaved_flag.begin(), m_row_to_col_interleaved_flag.end(), [](auto &i)
        //               {
        //                   i = counter % 2 == 0 ? true : FALSE;
        //                   ++counter;
        //               });
        // std::for_each(m_row_to_col_indices.begin(), m_row_to_col_indices.end(), [](auto &i)
        //               { i = i % 2 == 0 ? i : i - 1; });
        // std::iota(m_row_interleaved_indices.begin(), m_row_interleaved_indices.end(), 0);
        // std::iota(m_agg_indices.begin(), m_agg_indices.end(), 0);
        // std::iota(m_curr_map_indices.begin(), m_curr_map_indices.end(), 0);
        // std::for_each(m_agg_indices.begin(), m_agg_indices.end(), [&](const size_t &i) {
        //     size_t main = 0;
        //     if (i < m_dimension_agg)
        //     {
        //         main = i * 2;
        //     }
        //     else
        //     {
        //         main = i / m_dimension_agg * m_dimension_agg * 4 + (i - i / m_dimension_agg * m_dimension_agg) * 2;
        //     }
        //     m_curr_map_agg_indices[main] = i;
        //     m_curr_map_agg_indices[main + 1] = i;
        //     m_curr_map_agg_indices[main + m_dimension] = i;
        //     m_curr_map_agg_indices[main + m_dimension + 1] = i;
        // });

        for (auto i = 0; i < 4; i++) {
            m_curr_sub_maps.push_back(AllocateAligned<float>(m_npixels_agg));
        }
        /*for (auto i = 0; i < m_dimension * (m_dimension - 2); i += m_dimension)
        {
            for (auto j = 0; j < m_dimension_agg; j++)
            {
                m_curr_sub_maps_idx[0].push_back((i + j) * 2);
                m_curr_sub_maps_idx[1].push_back((i + j) * 2 + 1);
                m_curr_sub_maps_idx[2].push_back((i + j) * 2 + m_dimension);
                m_curr_sub_maps_idx[3].push_back((i + j) * 2 + m_dimension + 1);
            }
        }*/

        for (auto i = 0; i < 4; i++) {
            m_4sub_maps_ref.push_back(AllocateAligned<float*>(m_npixels_agg));
        }
        //assign references to the view and copy on-demand
        for (size_t i = 0; i < m_npixels_agg; ++i) {
            auto main = i * 2 + i / m_dimension_agg * m_dimension;
            m_4sub_maps_ref[0].get()[i] = &m_current_map.get()[main];
            m_4sub_maps_ref[1].get()[i] = &m_current_map.get()[main + 1];
            m_4sub_maps_ref[2].get()[i] = &m_current_map.get()[main + m_dimension];
            m_4sub_maps_ref[3].get()[i] = &m_current_map.get()[main + m_dimension + 1];
        }
    }

    void set_map(const uPtr_F& t_data_map)
    {
        std::copy(t_data_map.get(), t_data_map.get() + m_npixels, m_current_map.get());
        set_sub_maps();

        //set the row interleaved levels
        //set_row_interleaved();
    }

    void set_sub_maps()
    {
        if (m_dimension == 1)
            return;
        for (size_t i = 0; i < m_npixels_agg; i++) {
            m_curr_sub_maps[0].get()[i] = *m_4sub_maps_ref[0].get()[i];
            m_curr_sub_maps[1].get()[i] = *m_4sub_maps_ref[1].get()[i];
            m_curr_sub_maps[2].get()[i] = *m_4sub_maps_ref[2].get()[i];
            m_curr_sub_maps[3].get()[i] = *m_4sub_maps_ref[3].get()[i];
        }
    }

    uPtr_F& get_map()
    {
        return m_current_map;
    }

    void get_aggregate(uPtr_F& t_out_data_map, bool t_norm = FALSE)
    { //set the aggregate to t_data_map. Contains m_npixels/4 pixels
        if (m_dimension == 1) {
            throw(InvalidParams("There is no aggregate level below a 1x1 image!"));
        }

        if (m_npixels_agg >= Constants::get_max_lanes()) {
            //compute on lanes
            add_4(m_npixels_agg, m_curr_sub_maps[0], m_curr_sub_maps[1], m_curr_sub_maps[2], m_curr_sub_maps[3], t_out_data_map);
        } else {
            //else use the good old for loop
            for (auto i = 0; i < m_npixels_agg; ++i) {
                t_out_data_map.get()[i] = std::reduce(m_4sub_maps_ref.begin(), m_4sub_maps_ref.end(), 0.f, [&](auto a, auto b) { return *a.get()[i] + *b.get()[i]; });
            }
        }
        if (t_norm) {
            normalize_curr_map(t_out_data_map);
        }
    }

    template<typename D>
    float get_ngamma_sum(D d_func, float factor) const
    {
        //__PAR__
        for (size_t i = 0; i < m_npixels; ++i) {
            m_temp_storage.get()[i] = d_func(m_current_map.get()[i] + factor);
        }

        return Ops::reduce<tagF, float>(m_npixels, m_temp_storage);
    }

    void recompute_pixels()
    {
        //recompute on each submap
        for (auto& map : m_curr_sub_maps) {
            std::transform(execParUnseq, map.get(), map.get() + m_npixels_agg, map.get(), [&](const auto& i) {
                return rgamma(i + m_alpha, 1.f);
            });
        }

        //compute the aggregate
        std::fill(execParUnseq, m_temp_storage.get(), m_temp_storage.get() + m_npixels, 0.f);
        get_aggregate(m_temp_storage, true);

        //check for zeros
        //__PAR__
        for (size_t i = 0; i < m_npixels_agg; ++i) {
            if (std::fpclassify(m_temp_storage.get()[i]) == FP_ZERO) {
                float sum = 0.f;
                int counter = 0;
                while (sum == 0.0) {

                    for (auto& map : m_curr_sub_maps) {
                        map.get()[i] = rgamma(map.get()[i] + m_alpha, 1);
                        sum += map.get()[i];
                    }
                    ++counter;
                    if (counter > Constants::MAX_ITER_WHILE) {
                        throw(InconsistentData("Perhaps the number of input smoothing parameters is less than n (where nrows=2^n)?"));
                    }
                }
                m_temp_storage.get()[i] = sum;
            }
        }

        normalize_curr_map(m_temp_storage);
    };

    void set_total_exp_count(const float& ttlcnt_pr, const float& ttlcnt_exp)
    {
        if (m_dimension != 1) {
            throw(InvalidParams("Total exp count can only be set to the final level."));
        }
        m_current_map.get()[0] = rgamma(m_current_map.get()[0] + ttlcnt_pr, 1 / (1 + ttlcnt_exp));
    }

    float get_level_log_prior(bool t_flag = true)
    {
        //flag=true=>for use in comp_ms_prior
        //flag=FALSE=>for use in update_ms--log transforms the pixels
        float log_prior = 0.f;
        if (t_flag) {
            log_prior = std::transform_reduce(
              m_current_map.get(), m_current_map.get() + m_npixels, 0.f, [](const auto& i) { return log(i); }, std::plus<float>());

            return log_prior *= (m_alpha - 1);
        } else {
            //log transform all the pixels
            for (auto& map : m_curr_sub_maps) {
                std::transform(execParUnseq, map.get(), map.get() + m_npixels_agg, map.get(), [](const auto& i) {
                    return log(i);
                });
            }

            reset_temp_storage();

            //add the submaps
            get_aggregate(m_temp_storage);

            //get the log prior
            log_prior = Ops::reduce<tagF, float>(m_npixels_agg, m_temp_storage);

            log_prior *= m_alpha; /* Per Jason Kramer 13 Mar 2009 */

            return log_prior;
        }
    }

    void add_higher_agg(uPtr_F& t_higher_agg)
    {
        //add the higher agg to each of the submaps
        for (auto& map : m_curr_sub_maps) {
            Ops::v_op<f_vadd,tagF>(f_vadd(),m_npixels_agg,map,t_higher_agg,map);
        }

        //update the current submaps
        update_curr_map();
    }

    size_t get_dim()
    {
        return m_dimension;
    }

  protected:
    // uPtr_F m_row_interleaved_A;   //even rows of the current map
    // uPtr_F m_row_interleaved_B;   //odd rows of the current map
    // uPtr_F m_col_interleaved_A;   //even colums of the row interleaved sum
    // uPtr_F m_col_interleaved_B;   //odd rows of the row interleaved sum
    uPtr_F m_current_map;  //image of the current level
    uPtr_F m_temp_storage; //temporary storage with the current map dimensions
    //uPtr_F m_agg_norm_map; //
    //uPtr_F m_row_interleaved_sum; // the sum of even and odd rows. Has a size of~ dim/2 x dim
    /* Divide the current map into 4 sub maps. Their sum would be the aggregate matrix. Only do it if m_npixels/4 > max_lanes */
    std::vector<uPtr_F> m_curr_sub_maps;
    //std::vector<std::vector<size_t>> m_curr_sub_maps_idx;
    std::vector<uPtr_Fv> m_4sub_maps_ref;
    // std::vector<size_t> m_curr_sub_A_idx;
    // std::vector<size_t> m_curr_sub_B_idx;
    // std::vector<size_t> m_curr_sub_C_idx;
    // std::vector<size_t> m_curr_sub_D_idx;
    //bool m_is_lane_add{false};
    // size_t m_sub_npixels;
    size_t m_dimension;     // nrows or ncols of the current level
    size_t m_npixels;       //total number of pixels
    size_t m_npixels_agg;   //total number of pixels in a lower scale
    size_t m_dimension_agg; //half the current dimension
    //size_t m_half_npixels;  //total number of pixels in the row-interleaved image
    /*Pre-computed arrays*/
    //AlignedFreeUniquePtr<bool[]> m_row_to_col_interleaved_flag;
    // std::vector<bool> m_row_to_col_interleaved_flag; //alternating true false array to extract col interleaved pixels from the row-interleaved sum
    // std::vector<size_t> m_row_to_col_indices;        //every two  elements have the same value. Used to slice the row-interleaved sum
    // std::vector<size_t> m_row_interleaved_indices;   //0-to-m_half_npixels index array
    //std::vector<size_t> m_agg_indices;          //0-to-m_npixels_agg index array
    //std::vector<size_t> m_curr_map_agg_indices; //each element specifies its correspoinding index in the agg map
    //std::vector<size_t> m_curr_map_indices;     //0-to-m_npixels index array
    float& m_alpha;

  private:
    void reset_temp_storage()
    {
        std::fill(execParUnseq, m_temp_storage.get(), m_temp_storage.get() + m_npixels, 0.0f);
    }
    void add_4(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, const uPtr_F& t_c, const uPtr_F& t_d, uPtr_F& t_out)
    {
        const ScalableTag<float> a, b, c, d;
        const auto lanes = Constants::get_max_lanes();
        for (auto i = 0; i < t_npixels; i += lanes) {
            Store(
              (Load(a, t_a.get() + i) + Load(b, t_b.get() + i) + Load(c, t_c.get() + i) + Load(d, t_d.get() + i)), Constants::d, t_out.get() + i);
        }
    }

    void div(size_t t_npixels, const uPtr_Fv& t_a, const uPtr_F& t_b, uPtr_Fv& t_out)
    {
        const auto lanes = Constants::get_max_lanes();
        const ScalableTag<float> a, b;
        for (size_t i = 0; i < t_npixels; i += lanes) {
            Store(
              Div(Load(a, *t_a.get() + i), Load(b, t_b.get() + i)), a, *t_out.get() + i);
        }
    }

    void update_curr_map()
    {
        //__PAR__
        for (size_t i = 0; i < m_npixels_agg; i++) {
            *m_4sub_maps_ref[0].get()[i] = m_curr_sub_maps[0].get()[i];
            *m_4sub_maps_ref[1].get()[i] = m_curr_sub_maps[1].get()[i];
            *m_4sub_maps_ref[2].get()[i] = m_curr_sub_maps[2].get()[i];
            *m_4sub_maps_ref[3].get()[i] = m_curr_sub_maps[3].get()[i];
        }
    }
    // void get_agg_dim_gte_lanes(uPtr_F &t_out_data_map)
    // {

    //     add_4(m_npixels_agg, m_curr_sub_maps[0], m_curr_sub_maps[1], m_curr_sub_maps[2], m_curr_sub_maps[3], t_out_data_map);
    // }
    // void get_agg_dim_lt_lanes(uPtr_F &t_out_data_map)
    // {
    //     //the good old for loop without lanes
    //     for (auto i = 0; i < m_npixels_agg; ++i)
    //     {
    //         t_out_data_map.get()[i] = std::reduce(m_curr_sub_maps.begin(), m_curr_sub_maps.end(), 0.f, [&](auto a, auto b)
    //                                               { return a.get()[i] + b.get()[i]; });
    //     }
    // }

    // void generate_agg_norm_map(const uPtr_F &t_out_data_map)
    // {
    //     std::for_each(m_curr_map_indices.begin(), m_curr_map_indices.end(), [&](const auto &i) { m_agg_norm_map.get()[i] = t_out_data_map.get()[(m_curr_map_agg_indices[i]]; });
    // }

    void normalize_curr_map(uPtr_F& t_out_data_map)
    {
        //check for zeros in the aggregate
        //divide each pixel with its corresponding aggregated sum
        std::for_each(m_curr_sub_maps.begin(), m_curr_sub_maps.end(), [&](auto& a) {
            div(m_npixels_agg, a, t_out_data_map, a);
        });
        update_curr_map();
    }

    // void set_row_interleaved()
    // {
    //     for (size_t i = 0; i < m_dimension; i += 2)
    //     {
    //         std::copy_n(m_current_map.get() + i * m_dimension, m_dimension, m_row_interleaved_A.get() + i / 2 * m_dimension);

    //         std::copy_n(m_current_map.get() + (i + 1) * m_dimension, m_dimension, m_row_interleaved_B.get() + i / 2 * m_dimension);
    //     }
    // }

    // void set_col_interleaved()
    // {

    //     std::for_each(m_row_interleaved_indices.begin(), m_row_interleaved_indices.end(), [](const auto &i)
    //                   {
    //                       if (m_row_to_col_interleaved_flag[i])
    //                           m_col_interleaved_A.get()[m_row_to_col_indices[i]] = m_row_interleaved_sum.get()[i];
    //                       else
    //                           m_col_interleaved_B.get()[m_row_to_col_indices[i]] = m_row_interleaved_sum.get()[i];
    //                   });
    // }
};

class MultiScaleMap
{
  public:
    MultiScaleMap(CountsMap& t_obs)
    {

        //init alpha vector
        for (auto i = 0; i < m_nlevels; ++i)
            m_alpha.push_back(0.f);
    }

    MultiScaleLevelMap& get_level(int level)
    {
        return m_level_maps[level];
    }

    size_t get_nlevels()
    {
        return m_nlevels;
    }

    void set_alpha(int level, float t_value)
    {
        if (level > m_nlevels)
            throw(InvalidParams(std::string("The input level is greater than max level. Input level: ") + std::to_string(m_nlevels)));
        m_alpha[level] = t_value;
    }

    float get_alpha(int level)
    {
        if (level > m_nlevels)
            throw(InvalidParams(std::string("The input level is greater than max level. Input level: ") + std::to_string(m_nlevels)));
        return m_alpha[level];
    }

    void compute_cascade_agregations()
    {
        for (size_t level = 1; level <= m_nlevels; ++level) {
            auto& level_map_ref_cur = m_level_maps[level];
            auto& level_map_prev = m_level_maps[level - 1];

            level_map_prev.get_aggregate(level_map_ref_cur.get_map(), FALSE); //aggregate without normalizing
        }
    }

    void compute_cascade_proportions()
    {
        update_alpha_values();
        for (size_t level = 0; level < m_nlevels; ++level) {
            m_level_maps[level].recompute_pixels();
        }
    }

    void compute_cascade_log_scale_images()
    {
        //start from the highest aggregate and get to the final map
        for(auto i=m_nlevels;i>0;++i){
            m_level_maps[i-1].add_higher_agg(m_level_maps[i].get_map());
        }
    }

    float get_log_prior(float t_alpha, int power4)
    {
        if (power4 == 0) {
            return al_kap1 * log(t_alpha) + al_kap2 * pow(t_alpha, al_kap3);
        }
        if (power4 == 1) {
            return al_kap1 / t_alpha + al_kap2 * al_kap3 * pow(t_alpha, al_kap3 - 1.0);
        }
        if (power4 == 2) {
            al_kap1 / pow(t_alpha, 2.f) - al_kap2* al_kap3*(al_kap3 - 1.f) * pow(t_alpha, al_kap3 - 2.0f);
        }
    }

    void set_total_exp_count()
    {
        m_level_maps[m_nlevels].set_total_exp_count(m_ttlcnt_pr, m_ttlcnt_exp);
    }

    float get_max_agg_value()
    {
        return m_level_maps[m_nlevels].get_map().get()[0];
    }

    float get_ttlcnt_pr()
    {
        return m_ttlcnt_pr;
    }

    float get_ttlcnt_exp()
    {
        return m_ttlcnt_exp;
    }

    float get_log_prior_n_levels()
    {

        return std::reduce(m_level_maps.begin(), m_level_maps.end() - 1, 0.0f, [](const auto& a, const auto& b) {
            return a.get_level_log_prior(FALSE) + b.get_level_log_prior(FALSE);
        });
    }

  protected:
    size_t m_nlevels;
    std::vector<MultiScaleLevelMap> m_level_maps;
    float al_kap1, al_kap2, al_kap3, m_ttlcnt_pr, m_ttlcnt_exp;
    std::vector<float> m_alpha;
    void update_alpha_values()
    {
    }
    //std::vector
};

} // namespace HWY_NAMESPACE
} // namespace hwy

#endif