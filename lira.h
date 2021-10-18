#ifndef LIRA_H_
#define LIRA_H_

#include "hwy/foreach_target.h"

#include <R.h>
#include <R_ext/Random.h>
#define R_NO_REMAP 1
#include <Rinternals.h>
#define MATHLIB_STANDALONE 1
#include <Rmath.h>
#include <hwy/base.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <exception>
#include <execution>
#include <fstream>
#include <iomanip>
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
#include <iomanip>
#include <stdlib.h>
typedef std::stringstream sstr;

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

typedef AlignedFreeUniquePtr<size_t[]> uPtr_Int;
const auto& execParUnseq = std::execution::par_unseq;

typedef AlignedFreeUniquePtr<float[]> uPtr_F;
typedef AlignedFreeUniquePtr<float*[]> uPtr_Fv; //the "view" array
typedef Vec<ScalableTag<float>> vecF;
typedef ScalableTag<float> tagF;

struct controlType
{
    controlType(int t_iter, int t_max_iter, int t_burn, int t_save_iters, int t_save_thin, int t_fit_bkg_scl)
      : iter(t_iter)
      , max_iter(t_max_iter)
      , burn(t_burn)
      , save_iters(t_save_iters)
      , save_thin(t_save_thin)
      , fit_bkg_scl(t_fit_bkg_scl){};
    int iter;        /* the main iteration number */
    int max_iter;    /* the max number of iter for EM / number of Gibbs draws */
    int burn;        /* the number of burn-in draws to ignore when computing the posterior mean image */
    int save_iters;  /* 1 to print iters of src.img to R stream  & out file */
    int save_thin;   /* if save_iters, print every (save_thin)th to R & out file */
    int fit_bkg_scl; /* 1 to fit a scale parameter to the background model */
};

struct llikeType
{
    double cur{ 0 }; /* the current log likelihood of the iteration */
    double pre{ 0 }; /* the previous log likelihood of the iteration */
};

struct scalemodelType
{
    scalemodelType(float t_scale, float t_scale_pr, float t_scale_exp)
      : scale(t_scale)
      , scale_pr(t_scale_pr)
      , scale_exp(t_scale_exp)
    {}
    float scale;     /* the scale parameter */
    float scale_pr;  /* the prior on the total cnt in exposure = ttlcnt_exp */
    float scale_exp; /* the prior exposure in units of the actual exposure */
};

struct f_digamma
{
    float operator()(const float& v) { return digamma(v); }
};

struct f_trigamma
{
    float operator()(float v) { return trigamma(v); }
};

struct f_lgamma
{
    float operator()(float v) { return lgammafn(v); }
};

struct f_vadd
{
    vecF operator()(const vecF& t_a, const vecF& t_b) { return Add(t_a, t_b); }
};

class TempDS
{
  public:
    inline static std::map<std::string, uPtr_F> m_Fltv;
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
    IMG,
    BOTH
};

enum class DFunc
{
    DIGAMMA,
    TRIGAMMA
};

class Config
{
  public:
    Config(size_t t_max_iter, size_t t_burn, size_t t_save_iters, size_t t_save_thin, size_t t_fit_bkg_scl)
      : m_max_iter(t_max_iter)
      , m_burn(t_burn)
      , m_save_thin(t_save_thin)
      , m_save_iters(t_save_iters)
      , m_is_fit_bkg_scl(t_fit_bkg_scl)
    {
    }
    bool is_save() const { return iter % m_save_thin == 0 ? true : false; }
    size_t get_max_iter() const { return m_max_iter; }
    size_t get_save_thin() const { return m_save_thin; }
    size_t get_burn() const { return m_burn; }
    bool is_fit_bkg_scl() const { return m_is_fit_bkg_scl; }
    size_t iter{ 0 };
    bool is_save_post_mean() const { return iter > m_burn ? true : false; }
    size_t get_iter_m_burn() const { return iter - m_burn; }

  protected:
    size_t m_max_iter;
    size_t m_burn;
    size_t m_save_thin;
    size_t m_save_iters;
    bool m_is_fit_bkg_scl;
};

class Constants
{
  public:
    // clang-format off
                //http://www.graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
            //fancy way to compute the power of two
    inline static const int MultiplyDeBruijnBitPosition2[32]
    {
        0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
        8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
    };

    // clang-format on
    inline static const std::map<MapType, std::string> map_names = { { MapType::PSF, "PSF" }, { MapType::EXP, "Exposure" }, { MapType::COUNTS, "Counts" } };

    inline static const float convg_bisect{ 1e-8 };
    inline static const ScalableTag<float> d;
    typedef decltype(d) maxVec;
    inline static size_t get_max_lanes()
    {
        return Lanes(d);
    };
    inline static const float MH_sd_inflate{ 2.0 };
    inline static const int MH_iter{ 10 };
    inline static const int NR_END{ 1 };
    inline static const int MAX_ITER_WHILE{ 100 };

  protected:
};

class Utils
{
  public:
    static size_t get_power_2(size_t i);
    static size_t binary_roulette(size_t dim);

    template<class TagType, class T>
    inline static T reduce(const size_t& t_npixels, const uPtr_F& t_map)
    {
        const TagType a, d;
        auto sum_vec = Zero(a);
        auto const max_lanes = Constants::get_max_lanes();
        size_t i = 0;

        //__PAR__
        for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
            sum_vec = Load(a, t_map.get() + i) + sum_vec;
        }

        if (i < t_npixels) {
            sum_vec = IfThenElseZero(FirstN(d, t_npixels - i), Load(a, t_map.get() + i)) + sum_vec;
        }

        return static_cast<T>(GetLane(SumOfLanes(sum_vec)));
    }

    template<typename MaxTag, class Type>
    inline static void v_div(size_t t_npixels, const uPtr_F& t_a, Type& t_val, uPtr_F& t_result)
    {
        const MaxTag a, d;
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

    template<class MaxTag, typename ArithOp>
    inline static void v_op(ArithOp t_op, size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_result)
    {
        const MaxTag a, b, d;
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

    template<class VecType, class TagType>
    inline static void mul_add_pixels(const size_t& t_npixels, const size_t& max_lanes, const uPtr_F& t_a, const uPtr_F& t_b, VecType& t_result, size_t t_start_a = 0, size_t t_start_b = 0)
    {
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
        const TagType a, b, d;
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
            m_err_stream << "Incorrect " << t_map_name << " map dimensions: " << t_map_name << " map must be a square.\n";
            m_err_stream << " Input dimensions nrows: " << t_x << ", ncols: " << t_y;
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
    ImgMap(size_t t_nrows, size_t t_ncols, const MapType& t_map_type, std::string t_name)
      : m_nrows(t_nrows)
      , m_ncols(t_ncols)
      , m_map_type(t_map_type)
      , m_map_name(t_name)
      , m_npixels(m_nrows * m_ncols)
    {
        check_sizes();
        static_cast<T*>(this)->initialize();

        if (m_map_type != MapType::PSF) {
            m_power2 = Utils::get_power_2(m_nrows);
        }
    }

    ImgMap(const ImgMap&)=delete;
    ImgMap& operator=(const ImgMap&)=delete;

    size_t get_npixels() const
    {
        return m_npixels;
    }

    size_t get_dim() const
    {
        return m_nrows;
    }
    size_t get_power2()
    {
        return m_power2;
    }

  protected:
    const size_t m_nrows;
    const size_t m_ncols;
    size_t m_power2{ 1 };
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
            std::cout << "Warning: The PSF must have an odd dimension to allow the maximum to be exactly at the center of the image";
        }

        if (m_map_type != MapType::PSF && (m_nrows == 0 || (m_nrows & (m_nrows - 1)) || m_nrows < 8)) {
            throw(IncorrectDimensions(m_nrows, m_ncols, m_map_name, FALSE));
        }
    }
};

class WrappedIndexMap
{
  public:
    WrappedIndexMap(size_t& t_img_dim, size_t& t_psf_dim);

    const std::vector<size_t>& get_map() const;

    const size_t& get_npixels() const;
    const size_t& get_dim() const;
    const size_t& get_pad_dim() const;

  protected:
    size_t m_img_dim;
    size_t m_psf_dim;
    size_t m_pad_img_dim;
    size_t m_pad_size;             //Padding on each side of the image. Elements in the padded region will be wrapped
    size_t m_npixels;              //total pixels in the map inclusive of the padding
    std::vector<size_t> m_idx_map; //Each value maps the pixels from the input image to its padded/wrapped counterpart. The total padding is determined by the PSF's size.
  private:
    size_t wrap_idx(int t_idx, const size_t& t_dim);
};

class PSF : public ImgMap<PSF>
{
  public:
    PSF(size_t t_nrows, size_t t_ncols, const float* t_psf_mat, std::string t_name)
      : ImgMap<PSF>{ t_nrows, t_ncols, MapType::PSF, t_name }
      , m_mat_holder(t_psf_mat)
    {
    }

    PSF& initialize();

    const uPtr_F& get_rmat() const;
    uPtr_F& get_inv();
    void normalize_inv(float& sum);

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
  private:
    const float* m_mat_holder;
    size_t n_pixels{ 0 };
    float max_exp{ 1.f };
    size_t max_levels{ 1 };

  public:
    ExpMap(size_t t_nrows, size_t t_ncols, const float* t_expmap, std::string t_name)
      : m_mat_holder(t_expmap)
      , ImgMap<ExpMap>(t_nrows, t_ncols, MapType::EXP, t_name)
    {
    }
    void initialize();
    const uPtr_F& get_prod_map() const;
    float get_max_exp() const;
    const uPtr_F& get_map() const;

  protected:
    uPtr_F m_map;
    uPtr_F pr_det;
    uPtr_F m_prod;
};

class CountsMap : public ImgMap<CountsMap>
{
  public:
    CountsMap(size_t t_nrows, size_t t_ncols, CountsMapDataType t_map_data_type, std::string t_name, float* t_map = nullptr)
      : ImgMap(t_nrows, t_ncols, MapType::COUNTS, t_name)
      , m_map_holder(t_map)
      , map_data_type(t_map_data_type)
    {
    }

    CountsMap& initialize();
    void set_data_zero();
    void set_img_zero();
    uPtr_F& get_data_map();
    uPtr_F& get_img_map();
    void set_warped_mat(const WrappedIndexMap& t_w_idx_map, CountsMapDataType t_type = CountsMapDataType::DATA);
    void get_spin_data(uPtr_F& t_out, size_t spin_row, size_t spin_col);
    void set_spin_img(uPtr_F& t_in_map, size_t spin_row, size_t spin_col);
    uPtr_F& get_warped_img();
    uPtr_Fv& get_wmap_data_ref();
    size_t get_pad_dim() const;
    size_t get_wmap_dim() const;
    void re_norm_img(float t_val);

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

  private:
    const float* m_map_holder;
    std::vector<size_t> m_wmap_seq_idx;
    void check_wmap_set() const;
};

class MultiScaleLevelMap
{
  public:
    MultiScaleLevelMap(size_t t_dimension, float t_alpha);
    void set_map(const uPtr_F& t_data_map);
    void set_sub_maps();
    uPtr_F& get_map();
    void get_aggregate(uPtr_F& t_out_data_map, bool t_norm = FALSE);

    template<typename D>
    float get_ngamma_sum(D d_func, float factor) const
    {
        //__PAR__
        for (size_t i = 0; i < m_npixels; ++i) {
            m_temp_storage.get()[i] = d_func(m_current_map.get()[i] + factor);
        }

        return Utils::reduce<tagF, float>(m_npixels, m_temp_storage);
    };

    void recompute_pixels();
    void set_total_exp_count(const float& ttlcnt_pr, const float& ttlcnt_exp);
    float get_level_log_prior(bool t_flag = true);
    void add_higher_agg(uPtr_F& t_higher_agg);
    size_t get_dim();

  protected:
    uPtr_F m_current_map;  //image of the current level
    uPtr_F m_temp_storage; //temporary storage with the current map dimensions
    std::vector<uPtr_F> m_curr_sub_maps;
    std::vector<uPtr_Fv> m_4sub_maps_ref;
    size_t m_dimension;     // nrows or ncols of the current level
    size_t m_npixels;       //total number of pixels
    size_t m_npixels_agg;   //total number of pixels in a lower scale
    size_t m_dimension_agg; //half the current dimension
    float& m_alpha;

  private:
    void reset_temp_storage();
    void add_4(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, const uPtr_F& t_c, const uPtr_F& t_d, uPtr_F& t_out);
    void div(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_out);
    void update_curr_map();
    void normalize_curr_map(uPtr_F& t_out_data_map);
};

class MultiScaleMap
{
  public:
    MultiScaleMap(size_t t_power, float* t_alpha_vals, float t_kap1, float t_kap2, float t_kap3, float t_ttlcnt_pr, float t_ttlcnt_exp);
    MultiScaleLevelMap& get_level(int level);
    size_t get_nlevels() const;
    void set_alpha(size_t level, float t_value);
    float get_alpha(size_t level) const;
    void compute_cascade_agregations(bool t_norm);
    void compute_cascade_proportions();
    void compute_cascade_log_scale_images();
    float get_log_prior(float t_alpha, int power4);
    void set_total_exp_count();
    float get_max_agg_value();
    float get_ttlcnt_pr() const;
    float get_ttlcnt_exp() const;
    float get_al_kap2() const;
    float get_log_prior_n_levels(bool t_lt_flag /*True=> log transform pixels, False=> no log transform*/);
    void set_init_level_map(CountsMap& t_src);

  protected:
    size_t m_nlevels;
    std::vector<MultiScaleLevelMap> m_level_maps;
    float al_kap1, al_kap2, al_kap3, m_ttlcnt_pr, m_ttlcnt_exp;
    std::vector<float> m_alpha;
    void update_alpha_values()
    {
    }
};

template<class T>
class AsyncFileIO
{
  public:
    AsyncFileIO(const std::string t_out_file)
      : m_out_file_name(t_out_file)
    {
        m_out_file.open(m_out_file_name);
    }

    template<class I>
    AsyncFileIO& operator<<(const I& t_rhs)
    {
        m_out_file << t_rhs;
        return static_cast<T>(*this);
    }

  protected:
    std::string m_get_cmnt_str() { return "#\n# "; }
    std::string m_out_file_name;
    std::ofstream m_out_file;
};

class AsyncParamFileIO : public AsyncFileIO<AsyncParamFileIO>
{
  public:
    AsyncParamFileIO(std::string t_out_file, const ExpMap& t_exp_map, const MultiScaleMap& t_ms, const Config& t_conf);

    template<class I>
    AsyncParamFileIO& operator<<(const I& t_rhs)
    {
        m_out_file << std::left << std::setw(17) << std::setfill(' ') << t_rhs;
        return *this;
    }
};

class AsyncImgIO : public AsyncFileIO<AsyncImgIO>
{
  public:
    AsyncImgIO(std::string t_out_file)
      : AsyncFileIO(t_out_file)
    {}
    void write_img(CountsMap& t_map);
};

class Ops
{
  public:
    static float comp_ms_prior(CountsMap& t_src, MultiScaleMap& t_ms);
    static void redistribute_counts(PSF& t_psf, CountsMap& t_deblur, CountsMap& t_obs, llikeType& t_llike);

    static void remove_bkg_from_data(CountsMap& t_deblur, CountsMap& t_src, CountsMap& t_bkg, const scalemodelType& bkg_scale);

    static void add_cnts_2_adjust_4_exposure(const ExpMap& t_exp_map, CountsMap& t_src_map);

    static void check_monotone_convergence();

    static float update_image_ms(AsyncParamFileIO& t_param_file, const ExpMap& t_expmap, CountsMap& t_src, MultiScaleMap& t_ms, Config& t_config);
    static void update_alpha_ms(AsyncParamFileIO& t_out_file, MultiScaleMap& t_ms);

    static float update_alpha_ms_MH(const float& t_prop_mean, MultiScaleMap& t_ms, const size_t& t_level);

    static void update_scale_model(scalemodelType& t_scl_mdl, ExpMap& t_expmap, CountsMap& t_bkg_map);

    template<typename D, int Pow4>
    static float dnlpost_alpha(D t_dfunc, const float& t_alpha, MultiScaleMap& t_ms, const size_t& t_level)
    {
        auto factor = float(pow(4, Pow4));
        //D=digamma=>dlpost_alpha
        //D=trigamma=>ddlpost_alpha

        auto dim = t_ms.get_level(t_level).get_dim();
        float dlogpost = dim * dim * (factor * t_dfunc(4.0f * t_alpha) - 4.0f * t_dfunc(t_alpha));

        dlogpost += t_ms.get_level(t_level).get_ngamma_sum<D>(D(), t_alpha) - factor * t_ms.get_level(t_level + 1).get_ngamma_sum<D>(D(), 4 * t_alpha) + t_ms.get_log_prior(t_alpha, Pow4);

        return dlogpost;
    }

    static float dlpost_lalpha(float t_alpha, MultiScaleMap& t_ms, const size_t& t_level);

    static float ddlpost_lalpha(const float& t_alpha, MultiScaleMap& t_ms, const size_t& t_level);

    static float lpost_lalpha(float& t_alpha, MultiScaleMap& t_ms, const size_t& t_level);

    static void update_deblur_image(ExpMap& t_expmap, CountsMap& t_deblur, CountsMap& t_src, CountsMap& t_bkg, const scalemodelType& bkg_scale);
};

SEXP
image_analysis_R2(
  SEXP outmap,
  SEXP post_mean,
  SEXP cnt_vector,
  SEXP src_vector,
  SEXP psf_vector,
  SEXP map_vector,
  SEXP bkg_vector,
  SEXP out_filename,
  SEXP param_filename,
  SEXP max_iter,
  SEXP burn,
  SEXP save_iters,
  SEXP save_thin,
  SEXP nrow,
  SEXP ncol,
  SEXP nrow_psf,
  SEXP ncol_psf,
  SEXP em,
  SEXP fit_bkg_scl,
  SEXP alpha_init,
  SEXP alpha_init_len,
  SEXP ms_ttlcnt_pr,
  SEXP ms_ttlcnt_exp,
  SEXP ms_al_kap2,
  SEXP ms_al_kap1,
  SEXP ms_al_kap3

);

void
image_analysis_R(
  float* outmap,
  float* post_mean,
  float* cnt_vector,
  float* src_vector,
  float* psf_vector,
  float* map_vector,
  float* bkg_vector,
  char** out_filename,
  char** param_filename,
  int* max_iter,
  int* burn,
  int* save_iters,
  int* save_thin,
  int* nrow,
  int* ncol,
  int* nrow_psf,
  int* ncol_psf,
  int* em,
  int* fit_bkg_scl,
  float* alpha_init,
  int* alpha_init_len,
  float* ms_ttlcnt_pr,
  float* ms_ttlcnt_exp,
  float* ms_al_kap2,
  float* ms_al_kap1,
  float* ms_al_kap3);

void
bayes_image_analysis(
  float* t_outmap,
  float* t_post_mean,
  AsyncImgIO& t_out_file,
  AsyncParamFileIO& t_param_file,
  Config& t_conf,
  PSF& t_psf,
  ExpMap& t_expmap,
  CountsMap& t_obs,
  CountsMap& t_deblur,
  CountsMap& t_src,
  CountsMap& t_bkg,
  MultiScaleMap& t_ms,
  llikeType& llike,
  scalemodelType& bkg_scale);
};
}
HWY_AFTER_NAMESPACE();

#endif