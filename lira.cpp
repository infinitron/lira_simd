
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE _LIRA_SIMD_CPP_FILE_
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

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
#include <math.h>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

#include "hwy/aligned_allocator.h"
#include "hwy/base.h"
#include "hwy/highway.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <stdlib.h>
#include <typeinfo>
//#define VERBOSE

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

typedef std::stringstream sstr;
using prag_bayes_psf_func = std::function<double*(int)>;

template<class T>
void
clear(T i)
{
#ifdef VERBOSE
    std::cout << "Clear " << i << std::endl;
#endif
}

template<class uPtr_F>
void
display_image(const uPtr_F& img, size_t dim, int size = 5)
{
#ifdef VERBOSE
    for (auto i = 0; i < dim / size; i++) {
        for (auto j = 0; j < dim / size; ++j) {

            std::cout << img.get()[i * dim + j] << ' ';
        }
        std::cout << '\n';
    }
#endif
}

struct llikeType
{
    float cur{ 0.0 }; /* the current log likelihood of the iteration */
    float pre{ 0.0 }; /* the previous log likelihood of the iteration */
};

struct scalemodelType
{
    scalemodelType(float t_scale, float t_scale_pr, float t_scale_exp)
      : scale(t_scale)
      , scale_pr(t_scale_pr)
      , scale_exp(t_scale_exp)
    {}
    float scale;     /* the background scale parameter */
    float scale_pr;  /* the prior on the total cnt in exposure = ttlcnt_exp */
    float scale_exp; /* the prior exposure in units of the actual exposure */
};

struct f_digamma
{
    double operator()(const double v) { return digamma(v); }
    float operator()(const float v) { return digamma(v); }
};

struct f_trigamma
{
    double operator()(double v) { return trigamma(v); }
    float operator()(float v) { return trigamma(v); }
};

struct f_lgamma
{
    float operator()(float v) { return lgammafn(v); }
    double operator()(double v) { return lgammafn(v); }
};

template<class vecF>
struct f_vadd
{
    vecF operator()(const vecF& t_a, const vecF& t_b) { return Add(t_a, t_b); }
};

template<class uPtr_F>
class TempDS
{
  public:
    inline static std::map<std::string, uPtr_F> m_Fltv;
    inline static std::vector<int> m_PSF_inv_shuffle_idx;
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

class Config
{
  public:
    Config(size_t t_max_iter, size_t t_burn, size_t t_save_iters, size_t t_save_thin, size_t t_fit_bkg_scl, size_t t_is_prag_bayesian)
      : m_max_iter(t_max_iter)
      , m_burn(t_burn)
      , m_save_thin(t_save_thin)
      , m_save_iters(t_save_iters)
      , m_is_fit_bkg_scl(t_fit_bkg_scl)
      , m_is_psf_prag_bayesian_mode(t_is_prag_bayesian)
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
    bool is_psf_prag_bayesian() const { return m_is_psf_prag_bayesian_mode; }

  protected:
    size_t m_max_iter;
    size_t m_burn;
    size_t m_save_thin;
    size_t m_save_iters;
    bool m_is_fit_bkg_scl;
    bool m_is_psf_prag_bayesian_mode;
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

    inline static const double convg_bisect{ 1e-8 };
    inline static const double MH_sd_inflate{ 2.0 };
    inline static const int MH_iter{ 10 };
    inline static const int NR_END{ 1 };
    inline static const int MAX_ITER_WHILE{ 100 };                 //guards against infinite while loops
    inline static const auto& execParUnseq{ std::execution::seq }; //TODO: test why par_unseq leads to undefined behaviour!
    inline static const int N_ITER_PER_PSF{ 10 };                  //for prgaBayes type PSF generation
};

class Utils
{
  public:
    static size_t get_power_2(size_t i);
    static size_t binary_roulette(size_t dim); //random value between 0 and dim using a uniform dist

    /*Horizontally reduce an array*/
    template<class TagType, class T, class uPtr_F>
    static T reduce(const size_t& t_npixels, const uPtr_F& t_map);

    /*Vector division*/
    template<class MaxTag, class Type, class uPtr_F>
    static void v_div(size_t t_npixels, const uPtr_F& t_a, Type t_val, uPtr_F& t_result);

    /*Arbitrary vector operation on two arrays*/
    template<typename ArithOp, class uPtr_F, class MaxTag>
    static void v_op(ArithOp t_op, size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_result);

    /* C= A x B, reduce(C)*/
    template<class VecType, class TagType, class uPtr_F>
    inline static void mul_store_add_pixels(const size_t& t_npixels, const size_t& max_lanes, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_store, VecType& t_result, const size_t& t_start_a = 0, const size_t& t_start_b = 0, const size_t& t_start_store = 0);
};

/*************  Exception classes ************* */

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

    const std::string err_msg() const
    {
        return m_err_stream.str();
    }
};

class InvalidParams : virtual public std::exception
{
  protected:
    std::stringstream m_err_stream;

  public:
    explicit InvalidParams(const std::string t_msg)
    {
        m_err_stream << "Invalid parameters: " << t_msg;
    };

    explicit InvalidParams(const size_t& spin_row, const size_t& spin_col)
    {
        m_err_stream << "Invalid params: The spins can only be 0 or 1. spin_row(" << spin_row << ") spin_col(" << spin_col << ")\n";
    }

    virtual const char* what() const throw()
    {
        return m_err_stream.str().data();
    }

    const std::string err_msg() const
    {
        return m_err_stream.str();
    }
};

/*        Generic container base class for all the image-map types         */
/********  2D images will be implemented as 1D arrays     *******************/
class ImgMap
{
  public:
    ImgMap(size_t t_nrows, size_t t_ncols, const MapType& t_map_type, std::string t_name);

    ImgMap(const ImgMap&) = delete;
    ImgMap& operator=(const ImgMap&) = delete;

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
    size_t m_power2{ 1 }; //set to default if the size is not a power of two
    MapType m_map_type;
    std::string m_map_name;
    size_t m_npixels;

  private:
    void check_sizes(); //check the validity of the image dimensions for the specified map type
};

/* Creates a map with the dimensions of the image padded on both sides with half-size of the PSF. 
    Each pixel is mapped to pixels on the original image after wrapping. This requires the PSF to be odd-sized*/
class WrappedIndexMap
{
  public:
    WrappedIndexMap(size_t t_img_dim, size_t t_psf_dim);

    const std::vector<size_t>& get_map() const;

    const size_t& get_npixels() const;
    const size_t& get_dim() const;
    const size_t& get_pad_dim() const;

  protected:
    size_t m_img_dim;
    size_t m_psf_dim;
    size_t m_pad_img_dim;
    size_t m_pad_size; //Padding on each side of the image. Elements in the padded region will be wrapped
    size_t m_npixels;  //total pixels in the map inclusive of the padding
    std::vector<size_t> m_idx_map;

  private:
    size_t wrap_idx(int t_idx, const size_t& t_dim);
};

template<class uPtr_F, class tagF>
class PSF : public ImgMap
{
    using pix_type = typename std::pointer_traits<uPtr_F>::element_type;

  protected:
    uPtr_F m_mat{ nullptr };
    uPtr_F m_inv{ nullptr };
    uPtr_F m_rmat{ nullptr };                      //reversed PSF array for convolution, a.k.a 180 deg CCW rotated 2D matrix
    size_t m_L{ 0 }, m_R{ 0 }, m_U{ 0 }, m_D{ 0 }; //For compatibility with the previous code. Not implemented here.
    bool is_psf_prag_bayesian{ false };            //By default uses the same PSF for all the iterations
    const Config* m_config{ nullptr };
    pix_type m_min_psf; //minimum pixel value in the renormalized psf

  private:
    const double* m_mat_holder; //temporary storage before initialization checks
  public:
    PSF(size_t t_nrows, size_t t_ncols, const double* t_psf_mat, std::string t_name, const Config* t_config = nullptr)
      : ImgMap{ t_nrows, t_ncols, MapType::PSF, t_name }
      , m_config(t_config)
      , m_mat_holder(t_psf_mat)
    {
        if (m_config != nullptr) {
            is_psf_prag_bayesian = m_config->is_psf_prag_bayesian();
        }
        initialize();
    }

    PSF(const PSF&) = delete;
    PSF() = delete;
    PSF& operator=(const PSF&) = delete;

    PSF& initialize();

    const uPtr_F& get_rmat();
    uPtr_F& get_inv();
    void normalize_inv(pix_type sum);
    ~PSF()
    {        
        m_inv.release();
    }
};

template<class uPtr_F, class tagF>
class ExpMap : public ImgMap
{
    using pix_type = typename std::pointer_traits<uPtr_F>::element_type;

  private:
    const double* m_mat_holder;
    //size_t n_pixels{ 0 };
    pix_type max_exp{ 1.f };
    size_t max_levels{ 1 };

  public:
    ExpMap(size_t t_nrows, size_t t_ncols, const double* t_expmap, std::string t_name)
      : m_mat_holder(t_expmap)
      , ImgMap(t_nrows, t_ncols, MapType::EXP, t_name)
    {
        initialize();
    }
    void initialize();
    const uPtr_F& get_prod_map() const;
    auto get_max_exp() const;
    const uPtr_F& get_map() const;
    ExpMap(const ExpMap&) = delete;
    ExpMap& operator=(const ExpMap&) = delete;

  protected:
    uPtr_F m_map;
    uPtr_F pr_det;
    uPtr_F m_prod;
};

/***** Main class for the counts image type ****/
template<class uPtr_F, class uPtr_Fv>
class CountsMap : public ImgMap
{
    using pix_type = typename std::pointer_traits<uPtr_F>::element_type;

  public:
    CountsMap(size_t t_nrows, size_t t_ncols, CountsMapDataType t_map_data_type, std::string t_name, double* t_map = nullptr)
      : ImgMap(t_nrows, t_ncols, MapType::COUNTS, t_name)
      , m_map_holder(t_map)
      , map_data_type(t_map_data_type)
    {
        initialize();
    }

    CountsMap& initialize();
    void set_data_zero();
    void set_img_zero();
    uPtr_F& get_data_map();
    uPtr_F& get_img_map();
    void set_wraped_mat(const WrappedIndexMap& t_w_idx_map, CountsMapDataType t_type = CountsMapDataType::DATA);
    void get_spin_data(uPtr_F& t_out, size_t spin_row, size_t spin_col);
    void set_spin_img(uPtr_F& t_in_map, size_t spin_row, size_t spin_col);
    uPtr_F& get_wraped_img();
    uPtr_Fv& get_wmap_data_ref();
    size_t get_pad_dim() const; //dimensions of the padding
    size_t get_wmap_dim() const;
    template<class tagF>
    void re_norm_img(pix_type t_val);
    CountsMap(const CountsMap&) = delete;
    CountsMap& operator=(const CountsMap&) = delete;

  protected:
    uPtr_F m_data;                       //the counts
    uPtr_F m_img;                        //the image (expected counts)
    CountsMapDataType map_data_type;     //type of the data that should be read in
    uPtr_Fv m_wraped_img_ref{ nullptr }; //wraped (padded) img references
    uPtr_F m_wraped_img{ nullptr };
    uPtr_Fv m_wraped_data_ref{ nullptr };
    uPtr_F m_wraped_data{ nullptr };
    size_t m_npixels_wmap;
    bool m_is_wimg_set{ false };
    bool m_is_wdata_set{ false };
    size_t m_wmap_dim;
    size_t m_pad_dim{ 0 };

  private:
    const double* m_map_holder;
    std::vector<size_t> m_wmap_seq_idx;
    void check_wmap_img_set() const;
    void check_wmap_data_set() const;
};

/**** Similar to CountsMap but with aggregation and pixel-transformation capabilities ****/
/**** Each map is divided into four submaps whose sum gives a 4x4 pixel-aggregation of the original map ****/
template<class uPtr_F, class uPtr_Fv, class tagF>
class MultiScaleLevelMap
{
    using pix_type = typename std::pointer_traits<uPtr_F>::element_type;

  public:
    MultiScaleLevelMap(size_t t_dimension, pix_type t_alpha);
    void set_map(const uPtr_F& t_data_map);
    void set_sub_maps(); //from the current map
    uPtr_F& get_map();
    void get_aggregate(uPtr_F& t_out_data_map, bool t_norm = FALSE);
    template<typename D> //for use with dnlpost_alpha
    auto get_ngamma_sum(D d_func, pix_type factor) const
    {

        //d_func can be lgammafn, digamma, trigamma
        if (m_dimension > 1 && m_curr_sub_maps[0].get()[0] != m_current_map.get()[0]) {
            throw(std::string("The current map or sub maps are not up to date"));
        }
        //__PAR__
        for (size_t i = 0; i < m_npixels; ++i) {
            m_temp_storage.get()[i] = d_func(m_current_map.get()[i] + factor);
        }

        pix_type sum = Utils::reduce<tagF, pix_type>(m_npixels, m_temp_storage);

        return sum;
    };

    void recompute_pixels(); //m[i]=rgamma(1 + alpha,1)
    void set_total_exp_count(const pix_type ttlcnt_pr, const pix_type ttlcnt_exp);
    auto get_level_log_prior(bool t_flag = true);
    template<class vecF>
    void add_higher_agg(uPtr_F& t_higher_agg);
    size_t get_dim();
    auto get_alpha() const
    {
        return m_alpha;
    }
    void set_alpha(pix_type t_alpha)
    {
        m_alpha = t_alpha;
    }

  protected:
    uPtr_F m_current_map;  //image of the current level
    uPtr_F m_temp_storage; //temporary storage with the current map dimensions
    std::vector<uPtr_F> m_curr_sub_maps;
    std::vector<uPtr_Fv> m_4sub_maps_ref; //reference sub-maps for the current map. Can also use reference_wrapper<T>
    size_t m_dimension;                   // nrows or ncols of the current level
    size_t m_npixels;                     //total number of pixels
    size_t m_npixels_agg;                 //total number of pixels in a lower scale
    size_t m_dimension_agg;               //half the current dimension
    pix_type m_alpha;

  private:
    void update_curr_map(); //set the current map from the submaps
    void reset_temp_storage();
    void add_4(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, const uPtr_F& t_c, const uPtr_F& t_d, uPtr_F& t_out);
    void div(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_out);
    void normalize_curr_map(uPtr_F& t_out_data_map);
};

/*******  Container to hold maps at each aggregation level  *******/
/*******   Provides functions to transform pixels during aggregations *************/
template<class uPtr_F, class uPtr_Fv, class tagF>
class MultiScaleMap
{
    using pix_type = typename std::pointer_traits<uPtr_F>::element_type;

  public:
    MultiScaleMap(size_t t_power, double* t_alpha_vals, double t_kap1, double t_kap2, double t_kap3, double t_ttlcnt_pr, double t_ttlcnt_exp);
    MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>& get_level(int level);
    size_t get_nlevels() const;
    void set_alpha(size_t level, pix_type t_value);
    auto get_alpha(size_t level) const;
    void compute_cascade_agregations(bool t_norm = FALSE); //perform a 4x4 pixel aggregation on each level
    void compute_cascade_proportions();                    //recompute pixels in each aggregation

    template<class vecF>
    void compute_cascade_log_scale_images();          //adds the next level to aggregation after log transforming the current one
    auto get_log_prior(pix_type t_alpha, int power4); //refer to Esch et al. 2004 for the priors used here
    void set_total_exp_count();                       //on the highest agg level
    auto get_max_agg_value();
    auto get_ttlcnt_pr() const;
    auto get_ttlcnt_exp() const;
    auto get_al_kap2() const;
    auto get_log_prior_n_levels(bool t_lt_flag = FALSE /*True=> log transform pixels, False=> no log transform*/);
    void set_init_level_map(CountsMap<uPtr_F, uPtr_Fv>& t_src);
    void set_max_agg_value(pix_type val);

  protected:
    size_t m_nlevels;
    std::vector<MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>> m_level_maps;
    pix_type al_kap1, al_kap2, al_kap3, m_ttlcnt_pr, m_ttlcnt_exp;
    std::vector<pix_type> m_alpha;
};

/**** Base class for the file I/O (with no asynchronous I/O capabilities at present) *****/
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

template<class uPtr_F, class uPtr_Fv, class tagF>
class AsyncParamFileIO : public AsyncFileIO<AsyncParamFileIO<uPtr_F, uPtr_Fv, tagF>>
{
  public:
    AsyncParamFileIO(std::string t_out_file, const ExpMap<uPtr_F, tagF>& t_exp_map, const MultiScaleMap<uPtr_F, uPtr_Fv, tagF>& t_ms, const Config& t_conf);

    template<class I>
    AsyncParamFileIO& operator<<(const I& t_rhs)
    {
        this->m_out_file << std::right << std::setw(17) << std::setfill(' ') << t_rhs;
        return *this;
    }
};

template<class T, class Tv>
class AsyncImgIO : public AsyncFileIO<AsyncImgIO<T, Tv>>
{
  public:
    AsyncImgIO(std::string t_out_file)
      : AsyncFileIO<AsyncImgIO<T, Tv>>(t_out_file)
    {}
    void write_img(CountsMap<T, Tv>& t_map);
};

/******               LIRA's frontline warriors         ******/
/******      Class with all required worker functions   *****/
/****** With the same names as in the previous LIRA code ***/
class Ops
{
  public:
    template<class T, class Tv, class tagF>
    static double comp_ms_prior(CountsMap<T, Tv>& t_src, MultiScaleMap<T, Tv, tagF>& t_ms);

    template<class vecF, class T, class Tv, class tagF>
    static void redistribute_counts(PSF<T, tagF>& t_psf, CountsMap<T, Tv>& t_deblur, CountsMap<T, Tv>& t_obs, llikeType& t_llike);

    template<class tagF, class T, class Tv>
    static void remove_bkg_from_data(CountsMap<T, Tv>& t_deblur, CountsMap<T, Tv>& t_src, CountsMap<T, Tv>& t_bkg, const scalemodelType& bkg_scale);

    template<class T, class Tv, class tagF>
    static void add_cnts_2_adjust_4_exposure(const ExpMap<T, tagF>& t_exp_map, CountsMap<T, Tv>& t_src_map);

    template<class T, class Tv, class tagF>
    static void check_monotone_convergence(AsyncParamFileIO<T, Tv, tagF>& t_param_file, llikeType& t_llike, MultiScaleMap<T, Tv, tagF>& t_ms_map, Config& t_conf);

    template<class T, class Tv, class tagF, class vecF>
    static double update_image_ms(AsyncParamFileIO<T, Tv, tagF>& t_param_file, const ExpMap<T, tagF>& t_expmap, CountsMap<T, Tv>& t_src, MultiScaleMap<T, Tv, tagF>& t_ms, const Config& t_config);

    template<class T, class Tv, class tagF>
    static void update_alpha_ms(AsyncParamFileIO<T, Tv, tagF>& t_out_file, MultiScaleMap<T, Tv, tagF>& t_ms, const Config& t_config);

    template<class T, class Tv, class tagF>
    static double update_alpha_ms_MH(const double& t_prop_mean, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t& t_level);

    template<class vecF, class T, class Tv, class tagF>
    static void update_scale_model(scalemodelType& t_scl_mdl, ExpMap<T, tagF>& t_expmap, CountsMap<T, Tv>& t_bkg_map);

    template<typename D, int Pow4, class tagF, class T, class Tv>
    static double dnlpost_alpha(D t_dfunc, const double t_alpha, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t t_level);

    template<class T, class Tv, class tagF>
    static double dlpost_lalpha(double t_alpha, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t& t_level);

    template<class T, class Tv, class tagF>
    static double ddlpost_lalpha(const double& t_alpha, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t& t_level);

    template<class T, class Tv, class tagF>
    static double lpost_lalpha(double& t_alpha, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t& t_level);

    template<class vecF, class T, class Tv, class tagF>
    static void update_deblur_image(ExpMap<T, tagF>& t_expmap, CountsMap<T, Tv>& t_deblur, CountsMap<T, Tv>& t_src, CountsMap<T, Tv>& t_bkg, const scalemodelType& bkg_scale);
};

/** For future extensions with R **/
[[gnu::unused]] SEXP
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
  SEXP ms_al_kap3);

/** Main analysis function with in the HWY_NAMESPACE **/
template<class T, class Tv, class vecF, class tagF>
void
image_analysis_R(
  double* outmap,
  double* post_mean,
  double* cnt_vector,
  double* src_vector,
  double* psf_vector,
  double* map_vector,
  double* bkg_vector,
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
  double* alpha_init,
  int* alpha_init_len,
  double* ms_ttlcnt_pr,
  double* ms_ttlcnt_exp,
  double* ms_al_kap2,
  double* ms_al_kap1,
  double* ms_al_kap3,
  int *use_prag_bayes_psf,
  const prag_bayes_psf_func& t_psf_func);

template<class vecF, class tagF, class T, class Tv>
void
bayes_image_analysis(
  double* t_outmap,
  double* t_post_mean,
  AsyncImgIO<T, Tv>& t_out_file,
  AsyncParamFileIO<T, Tv, tagF>& t_param_file,
  Config& t_conf,
  PSF<T, tagF>& t_psf,
  ExpMap<T, tagF>& t_expmap,
  CountsMap<T, Tv>& t_obs,
  CountsMap<T, Tv>& t_deblur,
  CountsMap<T, Tv>& t_src,
  CountsMap<T, Tv>& t_bkg,
  MultiScaleMap<T, Tv, tagF>& t_ms,
  llikeType& llike,
  scalemodelType& bkg_scale,
  const prag_bayes_psf_func& t_psf_func);

/** To export the main analysis function outside HWY_NAMESPACE**/
void
image_analysis_R_export(
  double* t_outmap,
  double* t_post_mean,
  double* t_cnt_vector,
  double* t_src_vector,
  double* t_psf_vector,
  double* t_map_vector,
  double* t_bkg_vector,
  char** t_out_filename,
  char** t_param_filename,
  int* t_max_iter,
  int* t_burn,
  int* t_save_iters,
  int* t_save_thin,
  int* t_nrow,
  int* t_ncol,
  int* t_nrow_psf,
  int* t_ncol_psf,
  int* t_em,
  int* t_fit_bkg_scl,
  double* t_alpha_init,
  int* t_alpha_init_len,
  double* t_ms_ttlcnt_pr,
  double* t_ms_ttlcnt_exp,
  double* t_ms_al_kap2,
  double* t_ms_al_kap1,
  double* t_ms_al_kap3,
  int* t_use_float,
  int* is_psf_prag_bayesian,
  const prag_bayes_psf_func& t_psf_func);

}
}
HWY_AFTER_NAMESPACE();

/******* Class and function Definitions ********/

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
size_t
Utils::get_power_2(size_t i)
{
    if ((i & (i - 1)) == 0) {
        return Constants::MultiplyDeBruijnBitPosition2[(uint32_t)(i * 0x077CB531U) >> 27];
    } else
        throw(printf("%zu Not a power of 2!", i));

    return 0;
}

size_t
Utils::binary_roulette(size_t dim)
{
    return (size_t)(dim * runif(0, 1));
}

template<class TagType, class T, class uPtr_F>
T
Utils::reduce(const size_t& t_npixels, const uPtr_F& t_map)
{
    const TagType a, d;
    auto sum_vec = Zero(a);
    auto const max_lanes = Lanes(d);
    size_t i = 0;

    if (t_npixels < max_lanes) {
        using pix_type = typename std::pointer_traits<uPtr_F>::element_type;
        pix_type sum = 0;
        for (size_t i = 0; i < t_npixels; ++i) {
            sum += t_map.get()[i];
        }
        return sum;
    }

    //__PAR__
    for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
        sum_vec += Load(a, t_map.get() + i);
    }

    if (i < t_npixels) {
        sum_vec += IfThenZeroElse(FirstN(d, max_lanes - t_npixels + i), LoadU(a, t_map.get() + t_npixels - max_lanes));
    }

    return GetLane(SumOfLanes(sum_vec));
}

template<class MaxTag, class Type, class uPtr_F>
void
Utils::v_div(size_t t_npixels, const uPtr_F& t_a, Type t_val, uPtr_F& t_result)
{
    const MaxTag a, d;
    auto max_lanes = Lanes(d);
    const auto div_vec = Set(a, t_val);
    size_t i = 0;

    if (t_npixels < max_lanes) {
        for (size_t i = 0; i < t_npixels; ++i) {
            t_result.get()[i] = t_a.get()[i] / t_val;
        }
        return;
    }
    for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
        Store(Div(Load(a, t_a.get() + i), div_vec), a, t_result.get() + i);
    }
    if (i < t_npixels) {
        StoreU(Div(LoadU(a, t_a.get() + t_npixels - max_lanes), div_vec),
               a,
               t_result.get() + i);
    }
}

template<typename ArithOp, class uPtr_F, class MaxTag>
void
Utils::v_op(ArithOp t_op, size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_result)
{
    const MaxTag a, b, d;
    auto max_lanes = Lanes(d);

    size_t i = 0;

    if (t_npixels < max_lanes) {
        throw(InvalidParams(std::string("") + "The number of pixels is smaller than the lane size! npixels=" + std::to_string(t_npixels)));
    }

    //__PAR__
    for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {
        Store(t_op(Load(a, t_a.get() + i), Load(b, t_b.get() + i)), a, t_result.get() + i);
    }

    if (i < t_npixels && t_npixels > max_lanes) {
        auto j = t_npixels - max_lanes;

        StoreU(t_op(LoadU(a, t_a.get() + j), LoadU(b, t_b.get() + j)), a, t_result.get() + j);
    }
}

template<class VecType, class TagType, class uPtr_F>
void
Utils::mul_store_add_pixels(const size_t& t_npixels, const size_t& max_lanes, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_store, VecType& t_result, const size_t& t_start_a, const size_t& t_start_b, const size_t& t_start_store)
{
    const TagType a, b, d;
    VecType mul_value;
    size_t i = 0;
    for (i = 0; i + max_lanes <= t_npixels; i += max_lanes) {

        mul_value = Mul(LoadU(a, t_a.get() + i + t_start_a), LoadU(b, (t_b.get() + i + t_start_b)));
        StoreU(mul_value, a, t_store.get() + i + t_start_store);
        t_result += mul_value;
    }
    if (i < t_npixels) {
        auto j = t_npixels - max_lanes;
        mul_value = Mul(LoadU(a, t_a.get() + j + t_start_a), LoadU(b, t_b.get() + j + t_start_b));
        StoreU(mul_value, a, t_store.get() + j + t_start_store);
        t_result += IfThenZeroElse(FirstN(a, i - j), mul_value);
    }
}

ImgMap::ImgMap(size_t t_nrows, size_t t_ncols, const MapType& t_map_type, std::string t_name)
  : m_nrows(t_nrows)
  , m_ncols(t_ncols)
  , m_map_type(t_map_type)
  , m_map_name(t_name)
  , m_npixels(m_nrows * m_ncols)
{
    check_sizes();

    if (m_map_type != MapType::PSF) {
        m_power2 = Utils::get_power_2(m_nrows);
    }
}

void
ImgMap::check_sizes()
{
    if (m_nrows != m_ncols) {
        throw(IncorrectDimensions(m_nrows, m_ncols, m_map_name, true));
    }

    if (m_map_type == MapType::PSF && m_nrows % 2 == 0) {
        std::cout << "Warning: The PSF must have an odd dimension to allow the maximum to be exactly at the center of the image." << std::endl;
    }

    if (m_map_type != MapType::PSF && (m_nrows == 0 || (m_nrows & (m_nrows - 1)) || m_nrows < 8)) {
        throw(IncorrectDimensions(m_nrows, m_ncols, m_map_name, FALSE));
    }
}

WrappedIndexMap::WrappedIndexMap(size_t t_img_dim, size_t t_psf_dim)
  : m_img_dim(t_img_dim)
  , m_psf_dim(t_psf_dim)
{
    if (t_psf_dim % 2 == 0) {
        throw(InvalidParams("Wrapped Index map can only be created with an odd sized PSF."));
    }
    m_pad_size = size_t(m_psf_dim / 2); //assumes an odd sized PSF
    m_pad_img_dim = 2 * m_pad_size + m_img_dim;
    m_npixels = pow(m_pad_img_dim, 2);
    size_t _a, _b;

    for (size_t i = 0; i < m_pad_img_dim; ++i) {
        for (size_t j = 0; j < m_pad_img_dim; ++j) {
            //use this map to create a wraped image and simply multiply it with the psf matrix between the edges of its
            //corresponding image map

            m_idx_map.push_back(wrap_idx(i - m_pad_size, t_img_dim) * t_img_dim + wrap_idx(j - m_pad_size, t_img_dim));
        }
    }
}

const size_t&
WrappedIndexMap::get_npixels() const
{
    return m_npixels;
}

const size_t&
WrappedIndexMap::get_dim() const
{
    return m_pad_img_dim;
}

const size_t&
WrappedIndexMap::get_pad_dim() const
{
    return m_pad_size;
}

const std::vector<size_t>&
WrappedIndexMap::get_map() const
{
    return m_idx_map;
}

size_t
WrappedIndexMap::wrap_idx(int t_idx, const size_t& t_dim)
{
    if (t_idx >= 0 && t_idx < t_dim) {
        return t_idx;
    }

    while (t_idx < 0)
        t_idx += t_dim;
    return t_idx % t_dim;
}

int
wrap_idx(int t_idx, const size_t& t_dim)
{
    if (t_idx >= 0 && t_idx < t_dim) {
        return t_idx;
    }

    while (t_idx < 0)
        t_idx += t_dim;
    return t_idx % t_dim;
}

template<class uPtr_F, class tagF>
PSF<uPtr_F, tagF>&
PSF<uPtr_F, tagF>::initialize()
{

    m_mat = std::move(AllocateAligned<pix_type>(m_npixels));
    m_rmat = std::move(AllocateAligned<pix_type>(m_npixels));
    m_inv = std::move(AllocateAligned<pix_type>(m_npixels));

    std::copy(m_mat_holder, m_mat_holder + m_npixels, m_mat.get());

    //normalize the psf
    auto sum = std::reduce(m_mat.get(), m_mat.get() + m_npixels, 0.0, std::plus<pix_type>());
    std::transform(m_mat.get(), m_mat.get() + m_npixels, m_mat.get(), [&](auto a) { return a / sum; });

    //reverse the psf array for use in convolution
    std::reverse_copy(m_mat.get(), m_mat.get() + m_npixels, m_rmat.get());
    std::fill(m_inv.get(), m_inv.get() + m_npixels, 0);

    if (is_psf_prag_bayesian) { //rescale the original PSF with its min value and reverse it
        //TODO
    }
    return *this;
}

template<class uPtr_F, class tagF>
const uPtr_F&
PSF<uPtr_F, tagF>::get_rmat()
{
    if (is_psf_prag_bayesian && m_config->iter % Constants::N_ITER_PER_PSF == 0) {

        //TODO
    }
    return m_rmat;
}

template<class uPtr_F, class tagF>
uPtr_F&
PSF<uPtr_F, tagF>::get_inv()
{
    return m_inv;
}

template<class uPtr_F, class tagF>
void
PSF<uPtr_F, tagF>::normalize_inv(pix_type sum)
{
    using pix_type = typename std::pointer_traits<uPtr_F>::element_type;
    Utils::v_div<tagF, pix_type, uPtr_F>(m_npixels, m_inv, sum, m_inv);
}

template<class uPtr_F, class tagF>
void
ExpMap<uPtr_F, tagF>::initialize()
{
    m_map = AllocateAligned<pix_type>(m_npixels);
    m_prod = AllocateAligned<pix_type>(m_npixels);

    std::copy(m_mat_holder, m_mat_holder + m_npixels, m_map.get());
    max_exp = *std::max_element(Constants::execParUnseq, m_map.get(), m_map.get() + m_npixels);

    tagF d;
    const auto max_vector = Set(d, max_exp);

    //normalize the exposure map
    for (size_t i = 0; i < m_npixels; i += Lanes(d)) {
        const auto a = Load(d, m_map.get() + i);
        Store(Div(a, max_vector), d, m_map.get() + i);
    }

    //Because wrapping is the default behaviour => pr_det=1, hence prod=m_map
    std::copy(Constants::execParUnseq, m_map.get(), m_map.get() + m_npixels, m_prod.get());
}

template<class uPtr_F, class tagF>
auto
ExpMap<uPtr_F, tagF>::get_max_exp() const
{
    return max_exp;
}

template<class uPtr_F, class tagF>
const uPtr_F&
ExpMap<uPtr_F, tagF>::get_prod_map() const
{
    return m_prod;
}

template<class uPtr_F, class tagF>
const uPtr_F&
ExpMap<uPtr_F, tagF>::get_map() const
{
    return m_map;
}

template<class uPtr_F, class uPtr_Fv>
CountsMap<uPtr_F, uPtr_Fv>&
CountsMap<uPtr_F, uPtr_Fv>::initialize()
{
    m_data = AllocateAligned<pix_type>(m_npixels);
    m_img = AllocateAligned<pix_type>(m_npixels);

    if (m_map_holder == nullptr) {
        set_data_zero();
        set_img_zero();
        return *this;
    }

    if (map_data_type == CountsMapDataType::DATA) {
        std::copy(m_map_holder, m_map_holder + m_npixels, m_data.get());
        set_img_zero();
    } else if (map_data_type == CountsMapDataType::IMG) {
        std::copy(m_map_holder, m_map_holder + m_npixels, m_img.get());
        set_data_zero();
    } else {
        //pass
    }

    return *this;
}

template<class uPtr_F, class uPtr_Fv>
void
CountsMap<uPtr_F, uPtr_Fv>::set_data_zero()
{
    std::fill(m_data.get(), m_data.get() + m_npixels, 0);
}

template<class uPtr_F, class uPtr_Fv>
void
CountsMap<uPtr_F, uPtr_Fv>::set_img_zero()
{
    std::fill(m_img.get(), m_img.get() + m_npixels, 0);
}

template<class uPtr_F, class uPtr_Fv>
uPtr_F&
CountsMap<uPtr_F, uPtr_Fv>::get_data_map()
{
    return m_data;
}

template<class uPtr_F, class uPtr_Fv>
uPtr_F&
CountsMap<uPtr_F, uPtr_Fv>::get_img_map()
{
    return m_img;
}

template<class uPtr_F, class uPtr_Fv>
void
CountsMap<uPtr_F, uPtr_Fv>::set_wraped_mat(const WrappedIndexMap& t_w_idx_map, CountsMapDataType t_type)
{
    m_pad_dim = t_w_idx_map.get_pad_dim();
    m_npixels_wmap = t_w_idx_map.get_npixels();
    m_wmap_dim = t_w_idx_map.get_dim();
    const auto widx_map = t_w_idx_map.get_map();

    if (m_wmap_seq_idx.size() != m_npixels_wmap) {
        for (size_t i = 0; i < m_npixels_wmap; ++i)
            m_wmap_seq_idx.push_back(i);
    }

    if (t_type == CountsMapDataType::IMG) {
        m_is_wimg_set = true;
        if (m_wraped_img_ref == nullptr) {
            m_wraped_img_ref = AllocateAligned<pix_type*>(m_npixels_wmap);
        }
        if (m_wraped_img == nullptr) {
            m_wraped_img = AllocateAligned<pix_type>(m_npixels_wmap);
        }
        for (auto i : m_wmap_seq_idx) {
            m_wraped_img_ref.get()[i] = &m_img.get()[widx_map[i]];
        }
    } else {
        m_is_wdata_set = true;
        if (m_wraped_data_ref == nullptr) {
            m_wraped_data_ref = AllocateAligned<pix_type*>(m_npixels_wmap);
        }
        if (m_wraped_data == nullptr) {
            m_wraped_data = AllocateAligned<pix_type>(m_npixels_wmap);
        }
        for (auto i : m_wmap_seq_idx) {
            m_wraped_data_ref.get()[i] = &m_data.get()[widx_map[i]];
        }
    }
}

template<class uPtr_F, class uPtr_Fv>
void
CountsMap<uPtr_F, uPtr_Fv>::get_spin_data(uPtr_F& t_out, size_t spin_row, size_t spin_col)
{
    //__PAR__
    for (size_t i = 0; i < m_nrows; ++i) {
        for (size_t j = 0; j < m_ncols; ++j) {
            t_out.get()[i * m_ncols + j] = m_data.get()[((i + spin_row) % m_nrows) * m_ncols + (j + spin_col) % m_ncols];
        }
    }
};

template<class uPtr_F, class uPtr_Fv>
void
CountsMap<uPtr_F, uPtr_Fv>::set_spin_img(uPtr_F& t_in_map, size_t spin_row, size_t spin_col)
{
    //TODO: Figure out an efficient way to skip the mod operation

    //__PAR__
    for (size_t i = 0; i < m_nrows; ++i) {
        for (size_t j = 0; j < m_ncols; ++j) {
            m_img.get()[((i + spin_row) % m_nrows) * m_ncols + (j + spin_col) % m_ncols] = exp(t_in_map.get()[i * m_ncols + j]);
        }
    }
}

//returns a const reference to the wraped image map. This will be used to redisribute counts
template<class uPtr_F, class uPtr_Fv>
uPtr_F&
CountsMap<uPtr_F, uPtr_Fv>::get_wraped_img()
{
    check_wmap_img_set();
    for (auto i : m_wmap_seq_idx) {
        m_wraped_img.get()[i] = *(m_wraped_img_ref.get()[i]);
    }

    return m_wraped_img;
}

//returns a reference to the wraped data map. Used to update data map in the multinomial calculations
template<class uPtr_F, class uPtr_Fv>
uPtr_Fv&
CountsMap<uPtr_F, uPtr_Fv>::get_wmap_data_ref()
{
    check_wmap_data_set();
    return m_wraped_data_ref;
}

template<class uPtr_F, class uPtr_Fv>
size_t
CountsMap<uPtr_F, uPtr_Fv>::get_pad_dim() const
{
    if (!(m_is_wimg_set || m_is_wdata_set)) {
        throw(IncompleteInitialization(m_map_name, "The wraped image/data is not set yet. Call set_wraped_mat first."));
    }
    return m_pad_dim;
}

template<class uPtr_F, class uPtr_Fv>
size_t
CountsMap<uPtr_F, uPtr_Fv>::get_wmap_dim() const
{
    if (!(m_is_wimg_set || m_is_wdata_set)) {
        throw(IncompleteInitialization(m_map_name, "The wraped image/data is not set yet. Call set_wraped_mat first."));
    }
    return m_wmap_dim;
}

template<class uPtr_F, class uPtr_Fv>
void
CountsMap<uPtr_F, uPtr_Fv>::check_wmap_img_set() const
{
    if (!m_is_wimg_set) {
        throw(IncompleteInitialization(m_map_name, "The wraped image is not set yet. Call set_wraped_mat first."));
    }
}

template<class uPtr_F, class uPtr_Fv>
void
CountsMap<uPtr_F, uPtr_Fv>::check_wmap_data_set() const
{
    if (!m_is_wdata_set) {
        throw(IncompleteInitialization(m_map_name, "The wraped data is not set yet. Call set_wraped_mat first."));
    }
}

template<class uPtr_F, class uPtr_Fv>
template<class tagF>
void
CountsMap<uPtr_F, uPtr_Fv>::re_norm_img(typename CountsMap<uPtr_F, uPtr_Fv>::pix_type t_val)
{
    const tagF a, b, d;
    const auto multiplier_v = Set(b, t_val);
    auto max_lanes = Lanes(d);
    size_t i = 0;

    for (i = 0; i + max_lanes <= m_npixels; i += max_lanes) {
        Store(Mul(Load(a, m_img.get() + i), multiplier_v), a, m_img.get() + i);
    }
    if (i < m_npixels) { //the control should not show up here
        Store(IfThenElseZero(
                FirstN(d, m_npixels - i), Mul(Load(a, m_img.get() + i), multiplier_v)),
              a,
              m_img.get() + i);
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::MultiScaleLevelMap(size_t t_dimension, pix_type t_alpha)
  : m_dimension(t_dimension)
  , m_dimension_agg(m_dimension / 2)
  , m_alpha(t_alpha)
{
    m_npixels = pow(m_dimension, 2);
    m_npixels_agg = m_npixels / 4; //zero for a single pixel level
    m_current_map = AllocateAligned<pix_type>(m_npixels);
    m_temp_storage = AllocateAligned<pix_type>(m_npixels);

    std::fill(Constants::execParUnseq, m_current_map.get(), m_current_map.get() + m_npixels, 0.);
    std::fill(Constants::execParUnseq, m_temp_storage.get(), m_temp_storage.get() + m_npixels, 0.);

    if (m_npixels < 4)
        return; //no need to proceed further if it's just a single pixel

    for (auto i = 0; i < 4; i++) {
        m_curr_sub_maps.push_back(AllocateAligned<pix_type>(m_npixels_agg));
    }

    for (auto i = 0; i < 4; i++) {
        m_4sub_maps_ref.push_back(AllocateAligned<pix_type*>(m_npixels_agg));
    }
    //assign references to the view and copy on-demand into the submaps
    for (size_t i = 0; i < m_npixels_agg; ++i) {
        auto main = i * 2 + i / m_dimension_agg * m_dimension;
        m_4sub_maps_ref[0].get()[i] = &m_current_map.get()[main];
        m_4sub_maps_ref[1].get()[i] = &m_current_map.get()[main + 1];
        m_4sub_maps_ref[2].get()[i] = &m_current_map.get()[main + m_dimension];
        m_4sub_maps_ref[3].get()[i] = &m_current_map.get()[main + m_dimension + 1];
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::set_map(const uPtr_F& t_data_map)
{
    std::copy(t_data_map.get(), t_data_map.get() + m_npixels, m_current_map.get());
    set_sub_maps();
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::set_sub_maps()
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

template<class uPtr_F, class uPtr_Fv, class tagF>
uPtr_F&
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::get_map()
{
    return m_current_map;
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::get_aggregate(uPtr_F& t_out_data_map, bool t_norm)
{ //set the aggregate to t_data_map. Contains m_npixels/4 pixels
    if (m_dimension == 1) {
        throw(InvalidParams("There is no aggregate level below a 1x1 image!"));
    }
    tagF d;
    if (m_npixels_agg >= Lanes(d)) {
        //compute on lanes
        add_4(m_npixels_agg, m_curr_sub_maps[0], m_curr_sub_maps[1], m_curr_sub_maps[2], m_curr_sub_maps[3], t_out_data_map);
    } else {
        //else use the good old for loop
        // clear("enter");
        for (size_t i = 0; i < m_npixels_agg; ++i) {

            t_out_data_map.get()[i] = std::accumulate(m_4sub_maps_ref.begin(), m_4sub_maps_ref.end(), 0, [&](const auto& a, const auto& b) { return a + *(b.get()[i]); });
        }
    }
    if (t_norm) {
        normalize_curr_map(t_out_data_map);
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::recompute_pixels()
{

    if (m_dimension == 1) {

        throw(InvalidParams("No aggregation level below a single-pixel! in recompute_pixels()"));
    }
    //recompute on each submap
    //equivalent to map[i]=rgamma(map[i]+alpha,1)
    for (auto& map : m_curr_sub_maps) {
        std::transform(Constants::execParUnseq, map.get(), map.get() + m_npixels_agg, map.get(), [&](const auto& i) {
            return rgamma(i + m_alpha, 1.f);
        });
    }

    std::fill(Constants::execParUnseq, m_temp_storage.get(), m_temp_storage.get() + m_npixels, 0);
    get_aggregate(m_temp_storage, false);

    //__PAR__
    for (size_t i = 0; i < m_npixels_agg; ++i) {
        if (m_temp_storage.get()[i] == 0.0) {
            pix_type sum = 0;
            int counter = 0;
            while (sum == 0.0) {

                for (auto& map : m_curr_sub_maps) {
                    map.get()[i] = rgamma(map.get()[i] + m_alpha, 1);
                    sum += map.get()[i];
                }
                ++counter;
                if (counter > Constants::MAX_ITER_WHILE) {
                    throw(InconsistentData("Perhaps the number of input smoothing parameters is less than n (where nrows=2^n)? Or a smoothing parameter is zero"));
                }
            }
            m_temp_storage.get()[i] = sum;
        }
    }

    normalize_curr_map(m_temp_storage); //also copies pixels from submaps to the current map
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::set_total_exp_count(const pix_type ttlcnt_pr, const pix_type ttlcnt_exp)
{
    if (m_dimension != 1) {
        throw(InvalidParams("Total exp count can only be set to the final level."));
    }
    m_current_map.get()[0] = rgamma(m_current_map.get()[0] + ttlcnt_pr, 1 / (1 + ttlcnt_exp));
}

template<class uPtr_F, class uPtr_Fv, class tagF>
auto
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::get_level_log_prior(bool t_flag)
{
    if (m_dimension == 1) {

        throw(InvalidParams("No aggregation level below a single-pixel! in get_level_log_prior()"));
    }
    //flag=true=>for use in comp_ms_prior--log transforms the pixels without copying
    //flag=FALSE=>for use in update_ms--log transforms and stores the pixels in the submaps
    pix_type log_prior = 0;
    if (t_flag) {
        log_prior = std::accumulate(
          m_current_map.get(), m_current_map.get() + m_npixels, pix_type(0.0), [&](const auto a, const auto& b) { return a + log(b); });

        return log_prior *= (m_alpha - 1);
    } else {
        //log transform all the pixels
        for (auto& map : m_curr_sub_maps) {
            std::transform(Constants::execParUnseq, map.get(), map.get() + m_npixels_agg, map.get(), [](const auto& i) {
                return log(i);
            });
        }

        reset_temp_storage();

        //add the submaps
        get_aggregate(m_temp_storage);

        //get the log prior
        log_prior = Utils::reduce<tagF, pix_type>(m_npixels_agg, m_temp_storage);

        log_prior *= m_alpha; /* Per Jason Kramer 13 Mar 2009 */

        return log_prior;
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
template<class vecF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::add_higher_agg(uPtr_F& t_higher_agg)
{
    if (m_dimension == 1) {

        throw(InvalidParams("Invalid operation on a single-pixel! in add_higher_agg()"));
    }

    //add the higher agg to each of the submaps
    //the current submaps are log transformed while the higher aggregate's map isn't
    tagF d;
    auto max_lanes = Lanes(d);

    if (m_npixels_agg >= max_lanes) {
        for (auto& map : m_curr_sub_maps) {
            Utils::v_op<f_vadd<vecF>, uPtr_F, tagF>(f_vadd<vecF>(), m_npixels_agg, map, t_higher_agg, map);
        }
    } else {
        for (auto& map : m_curr_sub_maps) {
            std::transform(map.get(), map.get() + m_npixels_agg, t_higher_agg.get(), map.get(), std::plus<pix_type>());
        }
    }

    //update the current map with the new submaps
    update_curr_map();
}

template<class uPtr_F, class uPtr_Fv, class tagF>
size_t
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::get_dim()
{
    return m_dimension;
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::reset_temp_storage()
{
    std::fill(Constants::execParUnseq, m_temp_storage.get(), m_temp_storage.get() + m_npixels, 0.0);
}
template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::add_4(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, const uPtr_F& t_c, const uPtr_F& t_d, uPtr_F& t_out)
{
    //works only when t_npixels>=max_lanes
    const tagF a, b, c, d;
    const auto lanes = Lanes(d);
    for (size_t i = 0; i < t_npixels; i += lanes) {
        Store(
          (Load(a, t_a.get() + i) + Load(b, t_b.get() + i) + Load(c, t_c.get() + i) + Load(d, t_d.get() + i)), d, t_out.get() + i);
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::div(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_out)
{
    tagF d;
    const auto lanes = Lanes(d);
    if (t_npixels < lanes) {
        //do the good old for loop
        for (auto i = 0; i < t_npixels; ++i) {
            t_out.get()[i] = t_a.get()[i] / t_b.get()[i];
        }
        return;
    }
    const tagF a, b;
    for (size_t i = 0; i < t_npixels; i += lanes) {
        Store(
          Div(Load(a, t_a.get() + i), Load(b, t_b.get() + i)), a, t_out.get() + i);
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::update_curr_map()
{
    if (m_dimension == 1) {

        throw(InvalidParams("Invalid operation on a single-pixel! in update_curr_map()"));
    }

    //__PAR__
    for (size_t i = 0; i < m_npixels_agg; i++) {
        *m_4sub_maps_ref[0].get()[i] = m_curr_sub_maps[0].get()[i];
        *m_4sub_maps_ref[1].get()[i] = m_curr_sub_maps[1].get()[i];
        *m_4sub_maps_ref[2].get()[i] = m_curr_sub_maps[2].get()[i];
        *m_4sub_maps_ref[3].get()[i] = m_curr_sub_maps[3].get()[i];
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>::normalize_curr_map(uPtr_F& t_out_data_map)
{

    if (m_dimension == 1) {

        throw(InvalidParams("Invalid operation on a single-pixel! in normalize_curr_map()"));
    }
    //check for zeros in the aggregate
    //divide each pixel with its corresponding aggregated sum
    std::for_each(m_curr_sub_maps.begin(), m_curr_sub_maps.end(), [&](auto& a) {
        div(m_npixels_agg, a, t_out_data_map, a);
    });
    update_curr_map();
}

template<class uPtr_F, class uPtr_Fv, class tagF>
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::MultiScaleMap(size_t t_power2, double* t_alpha_vals, double t_kap1, double t_kap2, double t_kap3, double t_ttlcnt_pr, double t_ttlcnt_exp)
  : al_kap1(t_kap1)
  , al_kap2(t_kap2)
  , al_kap3(t_kap3)
  , m_ttlcnt_exp(t_ttlcnt_exp)
  , m_ttlcnt_pr(t_ttlcnt_pr)
{

    m_nlevels = t_power2 + 1;

    //init alpha
    for (size_t i = 0; i < m_nlevels - 1; ++i) {
        if (t_alpha_vals[i] == 0) {
            throw(InvalidParams("The smoothing parameters cannot be zero!"));
        }
        m_alpha.push_back(t_alpha_vals[i]);
    }
    m_alpha.push_back(0.0); //alpha on the 1-pixel level. unused anywhere in the code. For consistency with the level map class

    //init ms level maps
    for (size_t i = 1; i <= m_nlevels; ++i) {
        m_level_maps.push_back(std::move(MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>(pow(2, m_nlevels - i), m_alpha[i - 1])));
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::set_max_agg_value(pix_type val)
{
    m_level_maps[m_nlevels - 1].get_map().get()[0] = val;
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::set_init_level_map(CountsMap<uPtr_F, uPtr_Fv>& t_src)
{
    //set the 0th level with the src img
    m_level_maps[0].set_map(t_src.get_img_map());
}

template<class uPtr_F, class uPtr_Fv, class tagF>
MultiScaleLevelMap<uPtr_F, uPtr_Fv, tagF>&
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_level(int level)
{
    return m_level_maps[level];
}

template<class uPtr_F, class uPtr_Fv, class tagF>
size_t
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_nlevels() const
{
    return m_nlevels;
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::set_alpha(size_t level, pix_type t_value)
{
    if (level >= m_nlevels)
        throw(InvalidParams(std::string("The input level is greater than max level. Input level: ") + std::to_string(m_nlevels)));
    m_level_maps[level].set_alpha(t_value);
}

template<class uPtr_F, class uPtr_Fv, class tagF>
auto
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_alpha(size_t level) const
{
    if (level > m_nlevels)
        throw(InvalidParams(std::string("The input level is greater than max level. Input level: ") + std::to_string(m_nlevels)));
    return m_level_maps[level].get_alpha();
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::compute_cascade_agregations(bool t_norm)
{
    for (size_t level = 1; level < m_nlevels; ++level) {
        auto& level_map_next = m_level_maps[level];
        auto& level_map_prev = m_level_maps[level - 1];
        level_map_prev.get_aggregate(level_map_next.get_map(), t_norm);
        level_map_next.set_sub_maps();
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::compute_cascade_proportions()
{
    for (size_t level = 0; level < (m_nlevels - 1); ++level) {
        m_level_maps[level].recompute_pixels();
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
template<class vecF>
void
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::compute_cascade_log_scale_images()
{
    //start from the highest aggregate and get to the final map
    for (auto level = m_nlevels - 1; level > 0; --level) { /*m[level-1]=log(m[level-1])+m[level]*/
        m_level_maps[level - 1].template add_higher_agg<vecF>(m_level_maps[level].get_map());
    }
}

template<class uPtr_F, class uPtr_Fv, class tagF>
auto
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_log_prior(pix_type t_alpha, int t_nderiv)
{
    //p(alpha) = (delta * alpha)^kap1 * exp(-delta/kap3 * alpha^kap3)
    //The current patameterization uses kap2 = delta/kap3
    pix_type value;
    if (t_nderiv == 0) {
        value = al_kap1 * log(al_kap2 * al_kap3) + al_kap1 * log(t_alpha) - al_kap2 * pow(t_alpha, al_kap3);
    } else if (t_nderiv == 1) {
        value = al_kap1 / t_alpha - al_kap2 * al_kap3 * pow(t_alpha, al_kap3 - 1.0);
    } else if (t_nderiv == 2) {
        value = -al_kap1 / pow(t_alpha, 2.f) - al_kap2 * al_kap3 * (al_kap3 - 1.f) * pow(t_alpha, al_kap3 - 2.0);
    } else
        throw(InvalidParams(std::string("t_nderiv can only be 0,1,2, Input value: ") + std::to_string(t_nderiv)));

    return value;
}

template<class uPtr_F, class uPtr_Fv, class tagF>
void
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::set_total_exp_count()
{
    m_level_maps[m_nlevels - 1].set_total_exp_count(m_ttlcnt_pr, m_ttlcnt_exp);
}

template<class uPtr_F, class uPtr_Fv, class tagF>
auto
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_max_agg_value()
{
    return m_level_maps[m_nlevels - 1].get_map().get()[0];
}

template<class uPtr_F, class uPtr_Fv, class tagF>
auto
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_ttlcnt_pr() const
{
    return m_ttlcnt_pr;
}

template<class uPtr_F, class uPtr_Fv, class tagF>
auto
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_ttlcnt_exp() const
{
    return m_ttlcnt_exp;
}

template<class uPtr_F, class uPtr_Fv, class tagF>
auto
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_al_kap2() const
{
    return al_kap2;
}

template<class uPtr_F, class uPtr_Fv, class tagF>
auto
MultiScaleMap<uPtr_F, uPtr_Fv, tagF>::get_log_prior_n_levels(bool t_lt_flag /*Default FALSE. True=> log transform pixels without storing, False=> log transforms and stores*/)
{
    auto it = m_level_maps.begin();
    pix_type sum = 0.0;
    for (size_t i = 0; i < (m_nlevels - 1); ++i) {
        sum += it->get_level_log_prior(t_lt_flag);
        ++it;
    }

    return sum;
}

template<typename D, int Pow4, class tagF, class T, class Tv>
double
Ops::dnlpost_alpha(D t_dfunc, const double t_alpha, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t t_level)
{
    auto factor = double(pow(4, Pow4));
    //D=digamma=>dlpost_alpha
    //D=trigamma=>ddlpost_alpha

    double dim = t_ms.get_level(t_level + 1).get_dim();
    double dlogpost = dim * dim * (factor * t_dfunc(4.0 * t_alpha) - 4.0 * t_dfunc(t_alpha));

    dlogpost += t_ms.get_level(t_level).template get_ngamma_sum<D>(D(), t_alpha);                  //current level
    dlogpost -= factor * t_ms.get_level(t_level + 1).template get_ngamma_sum<D>(D(), 4 * t_alpha); //next leel
    dlogpost += t_ms.get_log_prior(t_alpha, Pow4);

    return dlogpost;
}

template<class vecF, class T, class Tv, class tagF>
void
Ops::update_deblur_image(ExpMap<T, tagF>& t_expmap, CountsMap<T, Tv>& t_deblur, CountsMap<T, Tv>& t_src, CountsMap<T, Tv>& t_bkg, const scalemodelType& t_bkg_scale)
{
    T& deblur_img = t_deblur.get_img_map();
    const T& src_img = t_src.get_img_map();
    const T& bkg_img = t_bkg.get_img_map();
    const T& exp_map = t_expmap.get_map();
    auto npixels = t_deblur.get_npixels();
    const tagF a, b, c, d;
    vecF bkg_scale_v = Set(a, t_bkg_scale.scale);
    const auto max_lanes = Lanes(d);
    size_t i = 0;
    for (i = 0; i + max_lanes <= npixels; i += max_lanes) {
        Store(
          Mul(                                                                             /* deblur_img= (bkg x bkg_scale + src) x exp */
              MulAdd(Load(a, bkg_img.get() + i), bkg_scale_v, Load(b, src_img.get() + i)), /*(bkg x bkg_scale + src)*/
              Load(c, exp_map.get() + i)),                                                 /* x exp */
          a,
          deblur_img.get() + i);
    }
    if (i < npixels) {
        auto j = npixels - i;
        Store(
          Mul(/* deblur_img= (bkg x bkg_scale + src) x exp */
              MulAdd(Load(a, bkg_img.get() + j), bkg_scale_v, Load(b, src_img.get() + j)),
              Load(c, exp_map.get() + j)),
          a,
          deblur_img.get() + j);
    }
}

template<class T, class Tv, class tagF>
double
Ops::comp_ms_prior(CountsMap<T, Tv>& t_src, MultiScaleMap<T, Tv, tagF>& t_ms)
{

    t_ms.set_init_level_map(t_src);

    t_ms.compute_cascade_agregations(true /*normalize the current map with its aggregated map*/);
    double log_prior = t_ms.get_log_prior_n_levels(true /*log transform the pixels without storing*/);
    log_prior += (t_ms.get_ttlcnt_pr() - 1) * log(t_ms.get_max_agg_value()) - t_ms.get_ttlcnt_exp() * t_ms.get_max_agg_value();
    return log_prior;
}

template<class vecF, class uPtr_F, class uPtr_Fv, class tagF>
void
Ops::redistribute_counts(PSF<uPtr_F, tagF>& t_psf, CountsMap<uPtr_F, uPtr_Fv>& t_deblur, CountsMap<uPtr_F, uPtr_Fv>& t_obs, llikeType& t_llike)
{
    t_deblur.set_data_zero();

    tagF d;
    const auto max_lanes = Lanes(d);

    const auto psf_dim = t_psf.get_dim();
    const auto psf_npixels = t_psf.get_npixels();
    const auto wmap_dim = t_deblur.get_wmap_dim();
    const auto deblur_pad_dim = t_deblur.get_pad_dim();
    const auto deblur_dim = t_deblur.get_dim();
    const auto& deblur_img = t_deblur.get_img_map();
    const auto obs_npixels = t_obs.get_npixels();
    const uPtr_F& deblur_wmap = t_deblur.get_wraped_img();
    const int D = psf_dim / 2;

    const uPtr_F& psf_rmat = t_psf.get_rmat();

    const uPtr_F& obs_map = t_obs.get_data_map();
    uPtr_Fv& deblur_wmap_data_ref = t_deblur.get_wmap_data_ref();
    uPtr_F& psf_inv = t_psf.get_inv();

    //loop over each pixel in the observation
    //The equivalent operation is multiplying the wraped deblur_img with the PSF matrix starting from the right-top edge
    size_t deblur_start_idx = 0;
    size_t psf_start_idx = 0;
    size_t obs_start_idx = 0;
    vecF sum_vec;
    double count = 0;
    double sum;
    double p_total = 1, p_cur;
    double sample;

    //setup the shuffler. Use the shuffler to randomize the redistribution sequence of counts in to the deblur_data
    if (TempDS<uPtr_F>::m_PSF_inv_shuffle_idx.size() != psf_npixels) {
        TempDS<uPtr_F>::m_PSF_inv_shuffle_idx.clear();
        for (size_t i = 0; i < psf_npixels; ++i) {
            TempDS<uPtr_F>::m_PSF_inv_shuffle_idx.push_back(i);
        }
    }

    for (size_t row = 0; row < deblur_dim; ++row) {
        for (size_t col = 0; col < deblur_dim; ++col) {
            deblur_start_idx = row * wmap_dim + col; //index for the wrapped/padded map
            obs_start_idx = row * deblur_dim + col;  //index for the unpadded map
            sum_vec = Zero(d);
            count = obs_map.get()[obs_start_idx];
            sum = 0;
            p_total = 1;
            //multiply with wraped image with the PSF centered at the current pixel
            for (size_t i = 0; i < psf_dim; ++i) { /*For each row in the psf*/
                Utils::mul_store_add_pixels<vecF, tagF>(psf_dim, max_lanes, deblur_wmap, psf_rmat, psf_inv, sum_vec, deblur_start_idx + i * wmap_dim, i * psf_dim, i * psf_dim);
            }

            sum = GetLane(SumOfLanes(sum_vec));

            if (sum == 0.0 && count > 0) {
                std::cout << "not allowed\n";
                return;
                throw(InconsistentData(row, col)); //psf doesn't allow data in this pixel
            }

            //update the log likelihood
            if (count > 0)
                t_llike.cur += obs_map.get()[obs_start_idx] * log(sum);
            t_llike.cur -= sum;

            t_psf.normalize_inv(sum);

            //Multinomial calculations
            //uncomment the following line to randomize the distribution for each pixel
            //std::random_shuffle(TempDS<uPtr_F>::m_PSF_inv_shuffle_idx.begin(),TempDS<uPtr_F>::m_PSF_inv_shuffle_idx.end());
            auto& shuffled_idx = TempDS<uPtr_F>::m_PSF_inv_shuffle_idx;

            for (auto& i : shuffled_idx) {
                if (psf_inv.get()[i] && count > 0.0) {
                    p_cur = psf_inv.get()[i] / p_total;
                    sample = (p_cur < 1.) ? rbinom(count, p_cur) : count;
                    count -= sample;
                } else
                    sample = 0;

                p_total -= psf_inv.get()[i];
                psf_inv.get()[i] = sample;
            } //multinomial calc loop

            //redistribute the counts to the deblur data map
            //*DO NOT* parallelize this loop! the deblur data map isn't an atomic type and multiple elements in the ref map point to the same location!
            for (size_t i = 0; i < psf_dim; ++i) {
                for (size_t j = 0; j < psf_dim; ++j) {
                    // if (i != 1 && j != 1 && psf_inv.get()[i * psf_dim + j] > 0) {
                    //     // clear("problem");
                    // }
                    *(deblur_wmap_data_ref.get()[deblur_start_idx + i * wmap_dim + j]) += psf_inv.get()[i * psf_dim + j];
                }
            }
        } //deblur col loop
    }     //deblur row loop
}

template<class tagF, class T, class Tv>
void
Ops::remove_bkg_from_data(CountsMap<T, Tv>& t_deblur, CountsMap<T, Tv>& t_src, CountsMap<T, Tv>& t_bkg, const scalemodelType& bkg_scale)
{
    using pix_type = typename std::pointer_traits<T>::element_type;

    auto npixels = t_src.get_npixels();
    auto img_dim = t_src.get_dim();

    tagF a, b, c, d;
    auto max_lanes = Lanes(d);
    if (TempDS<T>::m_Fltv.count("rem_bkg_prob_src") == 0) {
        TempDS<T>::m_Fltv["rem_bkg_prob_src"] = AllocateAligned<pix_type>(npixels);
    }

    auto& prob_src = TempDS<T>::m_Fltv["rem_bkg_prob_src"];
    std::fill(prob_src.get(), prob_src.get() + npixels, 0.0);

    auto& src_img = t_src.get_img_map();
    auto& src_data = t_src.get_data_map();
    auto& bkg_data = t_bkg.get_data_map();
    auto& bkg_img = t_bkg.get_img_map();
    auto& deblur_data = t_deblur.get_data_map();
    auto bkg_scale_vec = Set(d, pix_type(bkg_scale.scale));

    //__PAR__
    for (size_t i = 0; i < npixels; i += max_lanes) { //compute prob_src: src/(bkg_scale * bkg+src)
        Store(Load(a, src_img.get() + i) / (MulAdd(bkg_scale_vec, Load(b, bkg_img.get() + i), Load(c, src_img.get() + i))), d, prob_src.get() + i);
    }

    //__PAR__
    for (size_t i = 0; i < npixels; ++i) {
        if (prob_src.get()[i] < 1) {
            src_data.get()[i] = rbinom(deblur_data.get()[i], prob_src.get()[i]);
            bkg_data.get()[i] = deblur_data.get()[i] - src_data.get()[i];
        } else {
            src_data.get()[i] = deblur_data.get()[i];
            bkg_data.get()[i] = 0;
        }
    }
}

template<class T, class Tv, class tagF>
void
Ops::add_cnts_2_adjust_4_exposure(const ExpMap<T, tagF>& t_exp_map, CountsMap<T, Tv>& t_src_map)
{
    using pix_type = typename std::pointer_traits<T>::element_type;

    auto npixels = t_src_map.get_npixels();
    tagF a, b, c, d;
    auto max_lanes = Lanes(d);
    if (TempDS<T>::m_Fltv.count("exp_missing_count") == 0) {
        TempDS<T>::m_Fltv["exp_missing_count"] = AllocateAligned<pix_type>(npixels);
    }

    auto& exp_missing_count = TempDS<T>::m_Fltv["exp_missing_count"];
    //reset the temp DS
    std::fill(Constants::execParUnseq, exp_missing_count.get(), exp_missing_count.get() + npixels, 0);

    const auto& prod_map = t_exp_map.get_prod_map();
    const auto& counts_img = t_src_map.get_img_map();
    auto& counts_data = t_src_map.get_data_map();

    //compute expected number of missing counts
    //__PAR__
    for (size_t i = 0; i < npixels; i += max_lanes) {
        Store(/* exp_missing_count = (1-expmap.prod) x counts.img = -expmap.prod x counts.img + counts.img */
              NegMulAdd(Load(a, prod_map.get() + i), Load(b, counts_img.get() + i), Load(c, counts_img.get() + i)),
              d,
              exp_missing_count.get());
    }

    //__PAR__
    for (size_t i = 0; i < npixels; ++i) {
        exp_missing_count.get()[i] = rpois(exp_missing_count.get()[i]);
    }

    //add again
    //__PAR__
    for (size_t i = 0; i < npixels; i += max_lanes) { /* counts.data = counts.data + exp_missing_count */
        Store(Load(a, counts_data.get() + i) + Load(b, exp_missing_count.get() + i), d, counts_data.get() + i);
    }
}

template<class T, class Tv, class tagF>
void
Ops::check_monotone_convergence(AsyncParamFileIO<T, Tv, tagF>& t_param_file, llikeType& t_llike, MultiScaleMap<T, Tv, tagF>& t_ms_map, Config& t_conf)
{
    if (t_conf.is_save()) {
        t_param_file << t_llike.cur;
        if (t_conf.iter > 1) {
            t_param_file << t_llike.cur - t_llike.pre;
        } else {
            t_param_file << 0.0;
        }
    }

    t_llike.pre = t_llike.cur;
}

template<class T, class Tv, class tagF, class vecF>
double
Ops::update_image_ms(AsyncParamFileIO<T, Tv, tagF>& t_param_file, const ExpMap<T, tagF>& t_expmap, CountsMap<T, Tv>& t_src, MultiScaleMap<T, Tv, tagF>& t_ms, const Config& t_conf)
{

    using pix_type = typename std::pointer_traits<T>::element_type;
    auto dim = t_src.get_dim();
    int spin_row = Utils::binary_roulette(dim); //random integer between [0,dim)
    int spin_col = Utils::binary_roulette(dim);
    pix_type log_prior = 0.0;

    //_____FILE_IO_____//
    if (t_conf.is_save()) {
        t_param_file << spin_row << spin_col;
    }

    //copy src data into level 0
    t_src.get_spin_data(t_ms.get_level(0).get_map(), spin_row, spin_col);
    t_ms.get_level(0).set_sub_maps();

    t_ms.compute_cascade_agregations(false); //without normalization

    update_alpha_ms(t_param_file, t_ms, t_conf);

    t_ms.compute_cascade_proportions();

    t_ms.set_total_exp_count();

    //_____FILE_IO_____//
    if (t_conf.is_save())
        t_param_file << t_ms.get_max_agg_value();

    log_prior = (t_ms.get_ttlcnt_pr() - 1) * log(t_ms.get_max_agg_value()) - t_ms.get_ttlcnt_exp() * t_ms.get_max_agg_value();

    t_ms.set_max_agg_value(log(t_ms.get_max_agg_value()));

    //log transform all the submaps but not the main map. The log transformed values will be used in the next step
    log_prior += t_ms.get_log_prior_n_levels(false); ///logprior += alpha * log(m[i])

    //for each log-transformed submap, add its non-transformed aggregate
    //m[level-1][1..4] += m[level]
    t_ms.template compute_cascade_log_scale_images<vecF>();

    t_src.set_spin_img(t_ms.get_level(0).get_map(), spin_row, spin_col);

    return log_prior;
}

template<class T, class Tv, class tagF>
void
Ops::update_alpha_ms(AsyncParamFileIO<T, Tv, tagF>& t_param_file, MultiScaleMap<T, Tv, tagF>& t_ms, const Config& t_conf)
{
    double lower, //lower
      middle,     //middle
      upper,      //upper
      dl_lower,   //dl
      dl_upper,   //dl
      dl_middle;  //dl

    auto nlevels = t_ms.get_nlevels();

    for (size_t level = 0; level < (nlevels - 1); ++level) {
        //initialize lower and upper
        lower = 1.0;

        while (Ops::dlpost_lalpha<T, Tv, tagF>(lower, t_ms, level) < 0) {
            lower /= 2.0;
        }

        dl_lower = Ops::dlpost_lalpha<T, Tv, tagF>(lower, t_ms, level);

        upper = lower * 2.0;

        while (Ops::dlpost_lalpha<T, Tv, tagF>(upper, t_ms, level) > 0) {

            upper *= 2.0;
        }

        dl_upper = Ops::dlpost_lalpha<T, Tv, tagF>(upper, t_ms, level);

        while ((upper - lower) > Constants::convg_bisect) {
            middle = (lower + upper) / 2.0;
            dl_middle = Ops::dlpost_lalpha<T, Tv, tagF>(middle, t_ms, level);
            if (dl_middle > 0) {
                lower = middle;
                dl_lower = dl_middle;
            } else {
                upper = middle;
                dl_upper = dl_middle;
            }
        } //bisect loop
        auto new_alpha = update_alpha_ms_MH((upper + lower) / 2.0, t_ms, level);
        t_ms.set_alpha(level, new_alpha);

        //_____FILE IO_____//
        if (t_conf.is_save())
            t_param_file << t_ms.get_alpha(level);
    }
}

template<class T, class Tv, class tagF>
double
Ops::update_alpha_ms_MH(const double& t_prop_mean, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t& t_level)
{
    double proposal, //
      current,       //
      prop_sd,       //
      lg_prop_mn,    //
      log_ratio;     //

    lg_prop_mn = log(t_prop_mean);
    current = t_ms.get_alpha(t_level);

    /******** compute proposal std deviation ********/
    prop_sd = -ddlpost_lalpha<T, Tv, tagF>(t_prop_mean, t_ms, t_level);

    if (t_prop_mean * sqrt(prop_sd) < 1e-10) {
        throw(InconsistentData("Infinite MH proposal variance in update_alpha_ms_MH"));
    } else
        prop_sd = Constants::MH_sd_inflate / sqrt(prop_sd);

    for (auto i = 0; i < Constants::MH_iter; ++i) {
        proposal = rlnorm(lg_prop_mn, prop_sd); /* log normal proposal dist'n */
        log_ratio =
          lpost_lalpha<T, Tv, tagF>(proposal, t_ms, t_level) - lpost_lalpha<T, Tv, tagF>(current, t_ms, t_level) + dlnorm(current, lg_prop_mn, prop_sd, 1) - dlnorm(proposal, lg_prop_mn, prop_sd, 1);

        if (runif(0, 1) < exp(log_ratio))
            current = proposal;
    }
    return (current);
}

template<class T, class Tv, class tagF>
double
Ops::lpost_lalpha(double& t_alpha, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t& t_level)
{
    return dnlpost_alpha<f_lgamma, 0>(f_lgamma(), t_alpha, t_ms, t_level) + log(t_alpha);
}

template<class T, class Tv, class tagF>
double
Ops::dlpost_lalpha(double t_alpha, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t& t_level)
{
    return dnlpost_alpha<f_digamma, 1>(f_digamma(), t_alpha, t_ms, t_level) * t_alpha + 1.;
}

template<class T, class Tv, class tagF>
double
Ops::ddlpost_lalpha(const double& t_alpha, MultiScaleMap<T, Tv, tagF>& t_ms, const size_t& t_level)
{
    return dnlpost_alpha<f_digamma, 1>(f_digamma(), t_alpha, t_ms, t_level) * t_alpha + dnlpost_alpha<f_trigamma, 2>(f_trigamma(), t_alpha, t_ms, t_level) * t_alpha * t_alpha;
}

template<class vecF, class T, class Tv, class tagF>
void
Ops::update_scale_model(scalemodelType& t_scl_mdl, ExpMap<T, tagF>& t_expmap, CountsMap<T, Tv>& t_bkg_map)
{
    using pix_type = typename std::pointer_traits<T>::element_type;

    const auto npixels = t_expmap.get_npixels();
    if (TempDS<T>::m_Fltv.count("update_scale_model") == 0) {
        TempDS<T>::m_Fltv["update_scale_model"] = AllocateAligned<pix_type>(npixels);
    }

    auto& temp_v = TempDS<T>::m_Fltv["update_scale_model"];
    std::fill(Constants::execParUnseq, temp_v.get(), temp_v.get() + npixels, 0.0);
    tagF a;
    vecF total_exp_v = Set(a, 0.0);

    //total_exp=exp.map[i] x bkg.img[i]  /* Per Jason Kramer's correction 13 Mar */
    //TODO: This can be computed  once and use for the rest of the iterations
    Utils::mul_store_add_pixels<vecF, tagF>(npixels, Lanes(a), t_expmap.get_map(), t_bkg_map.get_img_map(), temp_v, total_exp_v);
    double total_exp = GetLane(SumOfLanes(total_exp_v));

    //total_cnt=summation<bkg.data>
    double total_cnt = Utils::reduce<tagF, pix_type>(npixels, t_bkg_map.get_data_map());

    t_scl_mdl.scale = rgamma(total_cnt + t_scl_mdl.scale_pr, 1 / (total_exp + t_scl_mdl.scale_exp));
}

template<class T, class Tv, class tagF>
AsyncParamFileIO<T, Tv, tagF>::AsyncParamFileIO(std::string t_out_file, const ExpMap<T, tagF>& t_exp_map, const MultiScaleMap<T, Tv, tagF>& t_ms, const Config& t_conf)
  : AsyncFileIO<AsyncParamFileIO<T, Tv, tagF>>(t_out_file)
{
    using pix_type = typename std::pointer_traits<T>::element_type;
    tagF d;
    //print the header
    this->m_out_file << this->m_get_cmnt_str() << "Code will run in posterior sampling mode.\n";
    if (t_conf.is_fit_bkg_scl())
        this->m_out_file << this->m_get_cmnt_str() << "A scale parameter will be fit to the bkg model.\n";
    this->m_out_file << this->m_get_cmnt_str() << "The total number of Gibbs draws is " << t_conf.get_max_iter() << "\n";
    this->m_out_file << this->m_get_cmnt_str() << "Every " << t_conf.get_save_thin() << "th draw will be saved\n";
    this->m_out_file << this->m_get_cmnt_str() << "The model will be fit using the Multi Scale Prior.\n";
    this->m_out_file << this->m_get_cmnt_str() << "The data matrix is " << t_exp_map.get_dim() << " by " << t_exp_map.get_dim() << "\n";
    this->m_out_file << this->m_get_cmnt_str() << "The data file should contain a  2^" << t_ms.get_nlevels() - 1 << " by 2^" << t_ms.get_nlevels() - 1 << " matrix of counts.\n";
    this->m_out_file << this->m_get_cmnt_str() << "Starting Values for the smoothing parameter (alpha):\n";
    for (size_t i = 0; i < t_ms.get_nlevels() - 1; ++i) {
        this->m_out_file << this->m_get_cmnt_str() << "Aggregation level: " << i << ",     alpha: " << t_ms.get_alpha(i)
                         << (i == 0 ? "  (Full data)" : i == t_ms.get_nlevels() - 2 ? "  (2x2 table)"
                                                                                    : "")
                         << "\n";
    }
    this->m_out_file << this->m_get_cmnt_str() << "The prior distribution on the total count from the multiscale component is\n";
    this->m_out_file << this->m_get_cmnt_str() << "Gamma(" << t_ms.get_ttlcnt_pr() << ", " << t_ms.get_ttlcnt_exp() << ")\n";
    this->m_out_file << this->m_get_cmnt_str() << "The hyper-prior smoothing parameter (kappa 2) is " << t_ms.get_al_kap2() << ".\n";
    this->m_out_file << this->m_get_cmnt_str() << "The working precision is:" << typeid(pix_type).name();
    this->m_out_file << this->m_get_cmnt_str() << "Max vector length at the working precision :" << Lanes(d) << "\n";

    if (t_conf.is_psf_prag_bayesian()) {
        this->m_out_file << this->m_get_cmnt_str() << "Enabling the PragBayesian mode for the PSF. A new PSF is sampled once in every " << std::to_string(Constants::N_ITER_PER_PSF) << " iterations \n\n";
    } else {
        this->m_out_file << "\n";
    }

    //print the table header
    *this
      << "Iteration"
      << "logPost"
      << "stepSize"
      << "cycleSpinRow"
      << "cycleSpinCol";
    for (size_t i = 0; i < t_ms.get_nlevels() - 1; ++i) {
        *this << "smoothingParam" + std::to_string(i);
    }
    *this << "expectedMSCounts"
          << "bkgScale";
}

template<class T, class Tv>
void
AsyncImgIO<T, Tv>::write_img(CountsMap<T, Tv>& t_map)
{
    auto dim = t_map.get_dim();
    auto npixels = t_map.get_npixels();
    const auto& img = t_map.get_img_map();
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            this->m_out_file << img.get()[i * dim + j] << " ";
        }
        this->m_out_file << "\n";
    }
}

template<class T, class Tv>
void
write_img(CountsMap<T, Tv>& map, int s = 5)
{
#ifdef VERBOSE
    std::ofstream out;
    auto dim = map.get_dim();
    auto& img = map.get_img_map();
    for (auto i = 0; i < dim / s; ++i) {
        for (auto j = 0; j < dim / s; ++j) {
            std::cout << int(img.get()[i * dim + j]) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
#endif
}

template<class T, class Tv>
void
write_data(CountsMap<T, Tv>& map, int s = 5)
{
#ifdef VERBOSE
    auto dim = map.get_dim();
    auto& img = map.get_data_map();
    for (auto i = 0; i < dim / s; ++i) {
        for (auto j = 0; j < dim / s; ++j) {
            std::cout << img.get()[i * dim + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
#endif
}

template<class T, class Tv, class vecF, class tagF>
void
image_analysis_R(
  double* t_outmap,
  double* t_post_mean,
  double* t_cnt_vector,
  double* t_src_vector,
  double* t_psf_vector,
  double* t_map_vector,
  double* t_bkg_vector,
  char** t_out_filename,
  char** t_param_filename,
  int* t_max_iter,
  int* t_burn,
  int* t_save_iters,
  int* t_save_thin,
  int* t_nrow,
  int* t_ncol,
  int* t_nrow_psf,
  int* t_ncol_psf,
  int* t_em,
  int* t_fit_bkg_scl,
  double* t_alpha_init,
  int* t_alpha_init_len,
  double* t_ms_ttlcnt_pr,
  double* t_ms_ttlcnt_exp,
  double* t_ms_al_kap2,
  double* t_ms_al_kap1,
  double* t_ms_al_kap3,
  int* t_is_psf_prag_bayesian,
  const prag_bayes_psf_func& t_psf_func)
{

    Config conf(*t_max_iter, *t_burn, *t_save_iters, *t_save_thin, *t_fit_bkg_scl, *t_is_psf_prag_bayesian);

    PSF<T, tagF> psf_map(*t_nrow_psf, *t_nrow_psf, t_psf_vector, "psf_map", &conf);

    /* The observed counts and model */
    CountsMap<T, Tv> obs_map(*t_nrow, *t_ncol, CountsMapDataType::DATA, "observed_map", t_cnt_vector);

    /* The source counts and model (starting values) */
    CountsMap<T, Tv> src_map(*t_nrow, *t_ncol, CountsMapDataType::IMG, "source_map", t_src_vector);

    /* The background counts and model */
    CountsMap<T, Tv> bkg_map(*t_nrow, *t_ncol, CountsMapDataType::IMG, "background_map", t_bkg_vector);

    /* The deblurred counts and model */
    CountsMap<T, Tv> deblur_map(*t_nrow, *t_ncol, CountsMapDataType::BOTH, "deblur_map");

    /* Initialize the wraped matrices for the deblur_map */
    WrappedIndexMap wmap(deblur_map.get_dim(), psf_map.get_dim());
    deblur_map.set_wraped_mat(wmap, CountsMapDataType::IMG);
    deblur_map.set_wraped_mat(wmap, CountsMapDataType::DATA);

    /* The exposure map */
    ExpMap<T, tagF> exp_map(*t_nrow, *t_ncol, t_map_vector, "exposure_map");
    llikeType llike;

    /* The multiscale map */
    MultiScaleMap<T, Tv, tagF> ms_map(obs_map.get_power2(), t_alpha_init, t_ms_al_kap1[0], t_ms_al_kap2[0], t_ms_al_kap3[0], *t_ms_ttlcnt_pr, *t_ms_ttlcnt_exp);

    /* Re-Norm bkg->img into same units as src->img [AVC Oct 2009] */
    /* bkg_img = bkg_img x exp.max_val */
    bkg_map.template re_norm_img<tagF>(exp_map.get_max_exp());
    scalemodelType bkg_scale(1.0 /*starting value*/, 0.001 /*prior value*/, 0.0 /*prior value*/);

    AsyncImgIO<T, Tv> out_img_file(*t_out_filename);
    AsyncParamFileIO<T, Tv, tagF> out_param_file(*t_param_filename, exp_map, ms_map, conf);
    try {
        {
            bayes_image_analysis<vecF>(t_outmap, t_post_mean, out_img_file, out_param_file, conf, psf_map, exp_map, obs_map, deblur_map, src_map, bkg_map, ms_map, llike, bkg_scale,t_psf_func);
        }
    } catch (const InvalidParams& e) {
        std::cout << "\n"
                  << e.err_msg() << std::endl;
        out_param_file << e.err_msg();
    } catch (const InconsistentData& e) {
        std::cout << "\n"
                  << e.err_msg() << std::endl;
        out_param_file << e.err_msg();
    } catch (const std::exception& e) {
        std::cout << "\n"
                  << e.what() << std::endl;
        out_param_file << e.what();
    } catch (const std::string& e) {
        std::cout << "\n"
                  << e;
        out_param_file << e;
    }
    std::cout<<"Done running\n";
}

template<class vecF, class tagF, class T, class Tv>
void
bayes_image_analysis(
  double* t_outmap,
  double* t_post_mean,
  AsyncImgIO<T, Tv>& t_out_file,
  AsyncParamFileIO<T, Tv, tagF>& t_param_file,
  Config& t_conf,
  PSF<T, tagF>& t_psf,
  ExpMap<T, tagF>& t_expmap,
  CountsMap<T, Tv>& t_obs,
  CountsMap<T, Tv>& t_deblur,
  CountsMap<T, Tv>& t_src,
  CountsMap<T, Tv>& t_bkg,
  MultiScaleMap<T, Tv, tagF>& t_ms,
  llikeType& t_llike,
  scalemodelType& t_bkg_scale,
  const prag_bayes_psf_func& t_psf_func)
{
    /* Initlaize the R Random seed */
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    set_seed(std::rand(), std::rand());
    size_t npixels = t_obs.get_npixels();
    //GetRNGstate();
    //compute_expmap() => because wrapping is the only behaviour pr=map
    t_llike.cur = Ops::comp_ms_prior(t_src, t_ms);

    Ops::update_deblur_image<vecF>(t_expmap, t_deblur, t_src, t_bkg, t_bkg_scale);

    /************************************/
    /************MAIN LOOP***************/
    /************************************/

    for (t_conf.iter = 1; t_conf.iter <= t_conf.get_max_iter(); t_conf.iter++) {
        if (t_conf.is_save()) {
            t_param_file << "\n"
                         << t_conf.iter;
        }
        const auto pg_psf = t_psf_func(t_conf.iter);

        /********************************************************************/
        /*********    REDISTRIBUTE obs.data to deblur.data            *******/
        /*********    SEPERATE deblur.data into src.data and bkg.data *******/
        /*********    IF NECESSARY add  cnts to src.data              *******/
        /********************************************************************/
        Ops::redistribute_counts<vecF>(t_psf, t_deblur, t_obs, t_llike);

        Ops::remove_bkg_from_data<tagF>(t_deblur, t_src, t_bkg, t_bkg_scale);

        Ops::add_cnts_2_adjust_4_exposure(t_expmap, t_src);

        //no monotone convergence checks -> MCMC
        //this will just write the current and step log posterior values to the param file
        Ops::check_monotone_convergence(t_param_file, t_llike, t_ms, t_conf);

        t_llike.cur = Ops::update_image_ms<T, Tv, tagF, vecF>(t_param_file, t_expmap, t_src, t_ms, t_conf);

        /***************************************************************/
        /*********          UPDATE BACKGROUND MODEL            *********/
        /***************************************************************/

        if (t_conf.is_fit_bkg_scl()) {
            /* ExpMap Added Per Jason Kramer 13 Mar 2009 */
            Ops::update_scale_model<vecF>(t_bkg_scale, t_expmap, t_bkg);
            if (t_conf.is_save()) {
                t_param_file << t_bkg_scale.scale;
            }
        } /* if fit background scale */

        Ops::update_deblur_image<vecF>(t_expmap, t_deblur, t_src, t_bkg, t_bkg_scale);
        // write_img(t_deblur,2);
        if (t_conf.is_save()) {
            t_out_file.write_img(t_src);
        }
        if (t_conf.is_save_post_mean()) {
            double m = t_conf.get_iter_m_burn(); //iter-mburn
            const auto& src_img = t_src.get_img_map();
            //__PAR__
            for (size_t i = 0; i < npixels; ++i) {
                t_post_mean[i] = ((m - 1.) * t_post_mean[i] + src_img.get()[i]) / m;
            }
        }
    }
    //PutRNGstate();
}

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

)
{ /*TODO*/
    return 0;
}

void
image_analysis_R_export(
  double* t_outmap,
  double* t_post_mean,
  double* t_cnt_vector,
  double* t_src_vector,
  double* t_psf_vector,
  double* t_map_vector,
  double* t_bkg_vector,
  char** t_out_filename,
  char** t_param_filename,
  int* t_max_iter,
  int* t_burn,
  int* t_save_iters,
  int* t_save_thin,
  int* t_nrow,
  int* t_ncol,
  int* t_nrow_psf,
  int* t_ncol_psf,
  int* t_em,
  int* t_fit_bkg_scl,
  double* t_alpha_init,
  int* t_alpha_init_len,
  double* t_ms_ttlcnt_pr,
  double* t_ms_ttlcnt_exp,
  double* t_ms_al_kap2,
  double* t_ms_al_kap1,
  double* t_ms_al_kap3,
  int* t_use_float,
  int* is_psf_prag_bayesian,
  const prag_bayes_psf_func& t_psf_func)
{
    t_psf_func(1000);
    if (t_use_float[0] == 1) {
        // std::cout<<"Using float\n";
        using T = float;
        using uPtr_F = AlignedFreeUniquePtr<T[]>;
        using uPtr_Fv = AlignedFreeUniquePtr<T*[]>;
        using vecF = Vec<ScalableTag<T>>;
        using tagF = ScalableTag<T>;

        TempDS<uPtr_F> d;

        image_analysis_R<uPtr_F, uPtr_Fv, vecF, tagF>(t_outmap, t_post_mean, t_cnt_vector, t_src_vector, t_psf_vector, t_map_vector, t_bkg_vector, t_out_filename, t_param_filename, t_max_iter, t_burn, t_save_iters, t_save_thin, t_nrow, t_ncol, t_nrow_psf, t_ncol_psf, t_em, t_fit_bkg_scl, t_alpha_init, t_alpha_init_len, t_ms_ttlcnt_pr, t_ms_ttlcnt_exp, t_ms_al_kap2, t_ms_al_kap1, t_ms_al_kap3, is_psf_prag_bayesian,t_psf_func);
    } else {

        using T = double;
        using uPtr_F = AlignedFreeUniquePtr<T[]>;
        using uPtr_Fv = AlignedFreeUniquePtr<T*[]>;
        using vecF = Vec<ScalableTag<T>>;
        using tagF = ScalableTag<T>;

        TempDS<uPtr_F> d;

        image_analysis_R<uPtr_F, uPtr_Fv, vecF, tagF>(t_outmap, t_post_mean, t_cnt_vector, t_src_vector, t_psf_vector, t_map_vector, t_bkg_vector, t_out_filename, t_param_filename, t_max_iter, t_burn, t_save_iters, t_save_thin, t_nrow, t_ncol, t_nrow_psf, t_ncol_psf, t_em, t_fit_bkg_scl, t_alpha_init, t_alpha_init_len, t_ms_ttlcnt_pr, t_ms_ttlcnt_exp, t_ms_al_kap2, t_ms_al_kap1, t_ms_al_kap3, is_psf_prag_bayesian,t_psf_func);
    }
}

}
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
using prag_bayes_psf_func = std::function<double*(int)>;

HWY_EXPORT(image_analysis_R_export);
extern "C"
{

    void image_analysis_lira(
      double* t_outmap,
      double* t_post_mean,
      double* t_cnt_vector,
      double* t_src_vector,
      double* t_psf_vector,
      double* t_map_vector,
      double* t_bkg_vector,
      char** t_out_filename,
      char** t_param_filename,
      int* t_max_iter,
      int* t_burn,
      int* t_save_iters,
      int* t_save_thin,
      int* t_nrow,
      int* t_ncol,
      int* t_nrow_psf,
      int* t_ncol_psf,
      int* t_em,
      int* t_fit_bkg_scl,
      double* t_alpha_init,
      int* t_alpha_init_len,
      double* t_ms_ttlcnt_pr,
      double* t_ms_ttlcnt_exp,
      double* t_ms_al_kap2,
      double* t_ms_al_kap1,
      double* t_ms_al_kap3,
      int* t_use_float,
      int* is_psf_prag_bayesian,
      const prag_bayes_psf_func& t_psf_func)
    {
        HWY_DYNAMIC_DISPATCH(image_analysis_R_export)
        (t_outmap, t_post_mean, t_cnt_vector, t_src_vector, t_psf_vector, t_map_vector, t_bkg_vector, t_out_filename, t_param_filename, t_max_iter, t_burn, t_save_iters, t_save_thin, t_nrow, t_ncol, t_nrow_psf, t_ncol_psf, t_em, t_fit_bkg_scl, t_alpha_init, t_alpha_init_len, t_ms_ttlcnt_pr, t_ms_ttlcnt_exp, t_ms_al_kap2, t_ms_al_kap1, t_ms_al_kap3, t_use_float, is_psf_prag_bayesian, t_psf_func);
    }
}
}
#endif

//compile command for R
//g++ -shared lira.cpp -lgtest -lpthread -lR -lRmath -lhwy -I/usr/local/include/ -I/usr/share/R/include/ -O2 -fPIC -o lira2.so -O2 -std=c++17 -ltbb
