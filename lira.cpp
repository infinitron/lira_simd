#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE __FILE__
#include "lira.h"

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
    return (size_t)dim * runif(0, 1);
}

WrappedIndexMap::WrappedIndexMap(size_t& t_img_dim, size_t& t_psf_dim)
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
    if (t_idx >= 0)
        return t_idx;

    while (t_idx < 0)
        t_idx += 1;
    return t_idx % t_dim;
}

PSF&
PSF::initialize()
{

    m_mat = AllocateAligned<float>(m_npixels);
    m_inv = AllocateAligned<float>(m_npixels);

    std::copy(execParUnseq, m_mat_holder, m_mat_holder + m_npixels, m_mat.get());
    std::reverse_copy(execParUnseq, m_mat_holder, m_mat_holder + m_npixels, m_rmat.get());
    std::fill(execParUnseq, m_mat.get() + m_orig_size, m_mat.get() + m_aligned_size, 0.f);

    return *this;
    //TODO--compute L,R,U,D
}

const uPtr_F&
PSF::get_rmat() const
{
    return m_rmat;
}

uPtr_F&
PSF::get_inv()
{
    return m_inv;
}

void
PSF::normalize_inv(float& sum)
{
    Utils::v_div<tagF, float>(m_npixels, m_inv, sum, m_inv);
}

void
ExpMap::initialize()
{
    m_map = AllocateAligned<float>(n_pixels);
    //pr_det = AllocateAligned<float>(n_pixels);
    m_prod = AllocateAligned<float>(n_pixels);

    std::copy(m_mat_holder, m_mat_holder + n_pixels, m_map.get());
    max_exp = *std::max_element(execParUnseq, m_map.get(), m_map.get() + n_pixels);

    tagF d;
    const auto max_vector = Set(d, max_exp);

    //normalize the exposure map
    for (size_t i = 0; i < n_pixels; i += Lanes(d)) {
        const auto a = Load(d, m_map.get() + i);
        Store(Div(a, max_vector), d, m_map.get() + i);
    }

    //Because wrapping is the enforced default behaviour => pr_det=1, hence prod=m_map
    std::copy(execParUnseq, m_map.get(), m_map.get() + n_pixels, m_prod.get());
}

float
ExpMap::get_max_exp() const
{
    return max_exp;
}

const uPtr_F&
ExpMap::get_prod_map() const
{
    return m_prod;
}

const uPtr_F&
ExpMap::get_map() const
{
    return m_map;
}

CountsMap&
CountsMap::initialize()
{
    m_data = AllocateAligned<float>(m_npixels);
    m_img = AllocateAligned<float>(m_npixels);

    if (m_map_holder == nullptr) {
        set_data_zero();
        set_img_zero();
        return *this;
    }

    if (map_data_type == CountsMapDataType::DATA) {
        std::copy(m_map_holder, m_map_holder + m_npixels, m_data.get());
        set_img_zero();
    } else {
        std::copy(m_map_holder, m_map_holder + m_npixels, m_img.get());
        set_data_zero();
    }

    return *this;
}

void
CountsMap::set_data_zero()
{
    std::fill(m_data.get(), m_data.get() + m_npixels, 0.f);
}

void
CountsMap::set_img_zero()
{
    std::fill(m_img.get(), m_img.get() + m_npixels, 0.f);
}

uPtr_F&
CountsMap::get_data_map()
{
    return m_data;
}

uPtr_F&
CountsMap::get_img_map()
{
    return m_img;
}

void
CountsMap::set_warped_mat(const WrappedIndexMap& t_w_idx_map, CountsMapDataType t_type)
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

void
CountsMap::get_spin_data(uPtr_F& t_out, size_t spin_row, size_t spin_col)
{
    //__PAR__
    for (size_t i = 0; i < m_nrows; ++i) {
        for (size_t j = 0; j < m_ncols; ++j) {
            t_out.get()[i * m_ncols + j] = m_data.get()[((i + spin_row) % m_nrows) * m_ncols + (j + spin_col) % m_ncols];
        }
    }
};

void
CountsMap::set_spin_img(uPtr_F& t_in_map, size_t spin_row, size_t spin_col)
{
    //TODO: Figure out an efficient way to skip the mod operation

    //__PAR__
    for (size_t i = 0; i < m_nrows; ++i) {
        for (size_t j = 0; j < m_ncols; ++j) {
            m_img.get()[((i + spin_row) % m_nrows) * m_ncols + (j + spin_col) % m_ncols] = exp(t_in_map.get()[i * m_ncols + j]);
        }
    }
}

//returns a const ref to the warped image map. This will be used to redisribute counts (i.e., convolving the map with the PSF)
uPtr_F&
CountsMap::get_warped_img()
{
    check_wmap_set();
    for (auto i : m_wmap_seq_idx) {
        m_warped_img.get()[i] = *m_warped_img_ref.get()[i];
    }

    return m_warped_img;
}

//returns a reference to the warped data map. Used to update data map in the multinomial calculations
uPtr_Fv&
CountsMap::get_wmap_data_ref()
{
    return m_warped_data_ref;
}

size_t
CountsMap::get_pad_dim() const
{
    return m_pad_dim;
}

size_t
CountsMap::get_wmap_dim() const
{
    check_wmap_set();
    return m_wmap_dim;
}

void
CountsMap::check_wmap_set() const
{
    if (!m_is_wimg_set) {
        throw(IncompleteInitialization(m_map_name, "The warped image is not set yet. Call set_warped_img first."));
    }
}

void
CountsMap::re_norm_img(float t_val)
{
    const tagF a, b, d;
    const auto multiplier_v = Set(b, t_val);
    auto max_lanes = Constants::get_max_lanes();
    size_t i = 0;

    for (i = 0; i + max_lanes <= m_npixels; i += max_lanes) {
        Store(Mul(Load(a, m_img.get() + i), multiplier_v), a, m_img.get() + i);
    }
    if (i < m_npixels) {
        Store(IfThenElseZero(
                FirstN(d, m_npixels - i), Mul(Load(a, m_img.get() + i), multiplier_v)),
              a,
              m_img.get() + i);
    }
}

// class MultiScaleLevelMap
// {
//   public:
MultiScaleLevelMap::MultiScaleLevelMap(size_t t_dimension, float t_alpha)
  : m_dimension(t_dimension)
  , m_dimension_agg(m_dimension / 2)
  , m_alpha(t_alpha)
{
    m_npixels = pow(t_dimension, 2);
    m_npixels_agg = m_npixels / 4; //zero for a single pixel level
    m_current_map = AllocateAligned<float>(m_npixels);
    m_temp_storage = AllocateAligned<float>(m_npixels);

    std::fill(execParUnseq, m_current_map.get(), m_current_map.get() + m_npixels, 0.f);
    std::fill(execParUnseq, m_temp_storage.get(), m_temp_storage.get() + m_npixels, 0.f);

    for (auto i = 0; i < 4; i++) {
        m_curr_sub_maps.push_back(AllocateAligned<float>(m_npixels_agg));
    }

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

void
MultiScaleLevelMap::set_map(const uPtr_F& t_data_map)
{
    std::copy(t_data_map.get(), t_data_map.get() + m_npixels, m_current_map.get());
    set_sub_maps();
}

void
MultiScaleLevelMap::set_sub_maps()
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

uPtr_F&
MultiScaleLevelMap::get_map()
{
    return m_current_map;
}

void
MultiScaleLevelMap::get_aggregate(uPtr_F& t_out_data_map, bool t_norm)
{ //set the aggregate to t_data_map. Contains m_npixels/4 pixels
    if (m_dimension == 1) {
        throw(InvalidParams("There is no aggregate level below a 1x1 image!"));
    }

    if (m_npixels_agg >= Constants::get_max_lanes()) {
        //compute on lanes
        add_4(m_npixels_agg, m_curr_sub_maps[0], m_curr_sub_maps[1], m_curr_sub_maps[2], m_curr_sub_maps[3], t_out_data_map);
    } else {
        //else use the good old for loop
        for (size_t i = 0; i < m_npixels_agg; ++i) {
            t_out_data_map.get()[i] = std::accumulate(m_4sub_maps_ref.begin(), m_4sub_maps_ref.end(), 0.f, [&](auto a, auto& b) { return a + *(b.get()[i]); });
        }
    }
    if (t_norm) {
        normalize_curr_map(t_out_data_map);
    }
}

void
MultiScaleLevelMap::recompute_pixels()
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
}

void
MultiScaleLevelMap::set_total_exp_count(const float& ttlcnt_pr, const float& ttlcnt_exp)
{
    if (m_dimension != 1) {
        throw(InvalidParams("Total exp count can only be set to the final level."));
    }
    m_current_map.get()[0] = rgamma(m_current_map.get()[0] + ttlcnt_pr, 1 / (1 + ttlcnt_exp));
}

float
MultiScaleLevelMap::get_level_log_prior(bool t_flag)
{
    //flag=true=>for use in comp_ms_prior--log transforms the pixels
    //flag=FALSE=>for use in update_ms
    float log_prior = 0.f;
    if (t_flag) {
        log_prior = std::accumulate(
          m_current_map.get(), m_current_map.get() + m_npixels, 0.f, [&](const auto a, auto b) { return a + log(b); });

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
        log_prior = Utils::reduce<tagF, float>(m_npixels_agg, m_temp_storage);

        log_prior *= m_alpha; /* Per Jason Kramer 13 Mar 2009 */

        return log_prior;
    }
}

void
MultiScaleLevelMap::add_higher_agg(uPtr_F& t_higher_agg)
{
    //add the higher agg to each of the submaps
    for (auto& map : m_curr_sub_maps) {
        Utils::v_op<tagF>(f_vadd(), m_npixels_agg, map, t_higher_agg, map);
    }

    //update the current submaps
    update_curr_map();
}

size_t
MultiScaleLevelMap::get_dim()
{
    return m_dimension;
}

//private:
void
MultiScaleLevelMap::reset_temp_storage()
{
    std::fill(execParUnseq, m_temp_storage.get(), m_temp_storage.get() + m_npixels, 0.0f);
}
void
MultiScaleLevelMap::add_4(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, const uPtr_F& t_c, const uPtr_F& t_d, uPtr_F& t_out)
{
    const tagF a, b, c, d;
    const auto lanes = Constants::get_max_lanes();
    for (size_t i = 0; i < t_npixels; i += lanes) {
        Store(
          (Load(a, t_a.get() + i) + Load(b, t_b.get() + i) + Load(c, t_c.get() + i) + Load(d, t_d.get() + i)), Constants::d, t_out.get() + i);
    }
}

void
MultiScaleLevelMap::div(size_t t_npixels, const uPtr_F& t_a, const uPtr_F& t_b, uPtr_F& t_out)
{
    const auto lanes = Constants::get_max_lanes();
    const tagF a, b;
    for (size_t i = 0; i < t_npixels; i += lanes) {
        Store(
          Div(Load(a, t_a.get() + i), Load(b, t_b.get() + i)), a, t_out.get() + i);
    }
}

void
MultiScaleLevelMap::update_curr_map()
{
    //__PAR__
    for (size_t i = 0; i < m_npixels_agg; i++) {
        *m_4sub_maps_ref[0].get()[i] = m_curr_sub_maps[0].get()[i];
        *m_4sub_maps_ref[1].get()[i] = m_curr_sub_maps[1].get()[i];
        *m_4sub_maps_ref[2].get()[i] = m_curr_sub_maps[2].get()[i];
        *m_4sub_maps_ref[3].get()[i] = m_curr_sub_maps[3].get()[i];
    }
}

void
MultiScaleLevelMap::normalize_curr_map(uPtr_F& t_out_data_map)
{
    //check for zeros in the aggregate
    //divide each pixel with its corresponding aggregated sum
    std::for_each(m_curr_sub_maps.begin(), m_curr_sub_maps.end(), [&](auto& a) {
        div(m_npixels_agg, a, t_out_data_map, a);
    });
    update_curr_map();
}

MultiScaleMap::MultiScaleMap(size_t t_power2, float* t_alpha_vals, float t_kap1, float t_kap2, float t_kap3, float t_ttlcnt_pr, float t_ttlcnt_exp)
  : al_kap1(t_kap1)
  , al_kap2(t_kap2)
  , al_kap3(t_kap3)
  , m_ttlcnt_exp(t_ttlcnt_exp)
  , m_ttlcnt_pr(t_ttlcnt_pr)
{

    m_nlevels = t_power2 + 1;

    //init alpha
    for (size_t i = 0; i < m_nlevels; ++i) {
        m_alpha.push_back(t_alpha_vals[i]);
    }
    m_alpha.push_back(0.0f); //unused anywhere in the code. For consistency with the level map class

    //init ms level maps
    for (size_t i = 0; i <= m_nlevels; ++i) {
        m_level_maps.push_back(MultiScaleLevelMap(pow(2, m_nlevels - 1), m_alpha[i]));
    }
}

void
MultiScaleMap::set_init_level_map(CountsMap& t_src)
{
    //set the 0th level with the src data
    m_level_maps[0].set_map(t_src.get_img_map());
}

MultiScaleLevelMap&
MultiScaleMap::get_level(int level)
{
    return m_level_maps[level];
}

size_t
MultiScaleMap::get_nlevels() const
{
    return m_nlevels;
}

void
MultiScaleMap::set_alpha(size_t level, float t_value)
{
    if (level > m_nlevels)
        throw(InvalidParams(std::string("The input level is greater than max level. Input level: ") + std::to_string(m_nlevels)));
    m_alpha[level] = t_value;
}

float
MultiScaleMap::get_alpha(size_t level) const
{
    if (level > m_nlevels)
        throw(InvalidParams(std::string("The input level is greater than max level. Input level: ") + std::to_string(m_nlevels)));
    return m_alpha[level];
}

void
MultiScaleMap::compute_cascade_agregations(bool t_norm = FALSE)
{
    for (size_t level = 1; level <= m_nlevels; ++level) {
        auto& level_map_ref_cur = m_level_maps[level];
        auto& level_map_prev = m_level_maps[level - 1];

        level_map_prev.get_aggregate(level_map_ref_cur.get_map(), t_norm);
    }
}

void
MultiScaleMap::compute_cascade_proportions()
{
    update_alpha_values();
    for (size_t level = 0; level < m_nlevels; ++level) {
        m_level_maps[level].recompute_pixels();
    }
}

void
MultiScaleMap::compute_cascade_log_scale_images()
{
    //start from the highest aggregate and get to the final map
    for (auto i = m_nlevels; i > 0; ++i) {
        m_level_maps[i - 1].add_higher_agg(m_level_maps[i].get_map());
    }
}

float
MultiScaleMap::get_log_prior(float t_alpha, int power4)
{
    if (power4 == 0) {
        return al_kap1 * log(t_alpha) + al_kap2 * pow(t_alpha, al_kap3);
    }
    if (power4 == 1) {
        return al_kap1 / t_alpha + al_kap2 * al_kap3 * pow(t_alpha, al_kap3 - 1.0);
    }
    if (power4 == 2) {
        return al_kap1 / pow(t_alpha, 2.f) - al_kap2 * al_kap3 * (al_kap3 - 1.f) * pow(t_alpha, al_kap3 - 2.0f);
    } else
        throw(printf("power4 can only be 0,1,2. Input value: %d", power4));
}

void
MultiScaleMap::set_total_exp_count()
{
    m_level_maps[m_nlevels].set_total_exp_count(m_ttlcnt_pr, m_ttlcnt_exp);
}

float
MultiScaleMap::get_max_agg_value()
{
    return m_level_maps[m_nlevels].get_map().get()[0];
}

float
MultiScaleMap::get_ttlcnt_pr() const
{
    return m_ttlcnt_pr;
}

float
MultiScaleMap::get_ttlcnt_exp() const
{
    return m_ttlcnt_exp;
}

float
MultiScaleMap::get_al_kap2() const
{
    return al_kap2;
}

float
MultiScaleMap::get_log_prior_n_levels(bool t_lt_flag = FALSE)
{

    return std::accumulate(m_level_maps.begin(), m_level_maps.end() - 1, 0.0f, [&](auto& a, auto& b) {
        return a + b.get_level_log_prior(t_lt_flag);
    });
}

void
update_deblur_image(ExpMap& t_expmap, CountsMap& t_deblur, CountsMap& t_src, CountsMap& t_bkg, const scalemodelType& t_bkg_scale)
{
    uPtr_F& deblur_img = t_deblur.get_img_map();
    const uPtr_F& src_img = t_src.get_img_map();
    const uPtr_F& bkg_img = t_bkg.get_img_map();
    const uPtr_F& exp_map = t_expmap.get_map();
    auto npixels = t_deblur.get_npixels();
    const tagF a, b, c, d;
    vecF bkg_scale_v = Set(a, t_bkg_scale.scale);
    const auto max_lanes = Constants::get_max_lanes();
    size_t i = 0;
    for (i = 0; i + max_lanes <= npixels; i += max_lanes) {
        Store(
          Mul(
            MulAdd(Load(a, bkg_img.get() + i), bkg_scale_v, Load(b, src_img.get() + i)), Load(c, exp_map.get() + i)),
          a,
          deblur_img.get() + i);
    }
    if (i < npixels) {
        auto mul_value = IfThenElseZero(
          FirstN(d, npixels - i),
          Mul(
            MulAdd(
              Load(a, bkg_img.get() + i), bkg_scale_v, Load(b, src_img.get() + i)),
            Load(c, exp_map.get() + i)));
        Store(mul_value, a, deblur_img.get() + i);
    }
}
float
Ops::comp_ms_prior(CountsMap& t_src, MultiScaleMap& t_ms)
{

    t_ms.set_init_level_map(t_src);
    t_ms.compute_cascade_agregations(TRUE /*normalize the current map with its aggregated map*/);
    float log_prior = t_ms.get_log_prior_n_levels(TRUE);
    log_prior += (t_ms.get_ttlcnt_pr() - 1) * log(t_ms.get_max_agg_value()) - t_ms.get_ttlcnt_exp() * t_ms.get_max_agg_value();
    return log_prior;
}

void
Ops::redistribute_counts(PSF& t_psf, CountsMap& t_deblur, CountsMap& t_obs, llikeType& t_llike)
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
                Utils::mul_store_add_pixels<vecF, tagF>(psf_dim, max_lanes, deblur_wmap, psf_rmat, psf_inv, sum_vec, ((row + i) * deblur_dim + col), i * psf_dim, i * psf_dim);
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

void
Ops::remove_bkg_from_data(CountsMap& t_deblur, CountsMap& t_src, CountsMap& t_bkg, const scalemodelType& bkg_scale)
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

void
Ops::add_cnts_2_adjust_4_exposure(const ExpMap& t_exp_map, CountsMap& t_src_map)
{
    auto npixels = t_src_map.get_npixels();
    auto max_lanes = Constants::get_max_lanes();
    tagF a, b, c, d;
    if (TempDS::m_Fltv.count("exp_missing_count") == 0) {
        TempDS::m_Fltv["exp_missing_count"] = AllocateAligned<float>(npixels);
    }

    const auto& prod_map = t_exp_map.get_prod_map();
    const auto& counts_img = t_src_map.get_img_map();
    auto& counts_data = t_src_map.get_data_map();
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

void
Ops::check_monotone_convergence()
{}

float
Ops::update_image_ms(AsyncParamFileIO& t_param_file, const ExpMap& t_expmap, CountsMap& t_src, MultiScaleMap& t_ms, Config& t_conf)
{
    auto dim = t_src.get_dim();
    int spin_row = dim * Utils::binary_roulette(dim);
    int spin_col = dim * Utils::binary_roulette(dim);
    float log_prior = 0.0f;

    //_____FILE_IO_____//
    if (t_conf.is_save()) {
        t_param_file << spin_row << spin_col;
    }

    //copy src data into level 0
    t_src.get_spin_data(t_ms.get_level(0).get_map(), spin_row, spin_col);

    //compute_aggreggations
    t_ms.compute_cascade_agregations();

    //update alpha
    update_alpha_ms(t_param_file, t_ms);

    //compute proportions
    t_ms.compute_cascade_proportions();

    //update the expected total count
    t_ms.set_total_exp_count();

    //_____FILE_IO_____//
    t_param_file << t_ms.get_max_agg_value();

    //compute log prior
    log_prior = (t_ms.get_ttlcnt_pr() - 1) * log(t_ms.get_max_agg_value()) - t_ms.get_ttlcnt_exp() * t_ms.get_max_agg_value();

    log_prior += t_ms.get_log_prior_n_levels();

    //compute the log scale image
    t_ms.compute_cascade_log_scale_images();

    //store new image in src.img
    t_src.set_spin_img(t_ms.get_level(0).get_map(), spin_row, spin_col);

    return log_prior;
}

void
Ops::update_alpha_ms(AsyncParamFileIO& t_param_file, MultiScaleMap& t_ms)
{
    float lower, //lower
      middle,    //middle
      upper,     //upper
      dl_lower,  //dl
      dl_upper,  //dl
      dl_middle; //dl

    auto nlevels = t_ms.get_nlevels();

    for (size_t level = 0; level < nlevels; ++level) {
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
        t_param_file << t_ms.get_alpha(level);
    }
}

float
Ops::update_alpha_ms_MH(const float& t_prop_mean, MultiScaleMap& t_ms, const size_t& t_level)
{
    float proposal, //
      current,      //
      prop_sd,      //
      lg_prop_mn,   //
      log_ratio;    //

    lg_prop_mn = log(t_prop_mean);
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
}

float
Ops::dlpost_lalpha(float t_alpha, MultiScaleMap& t_ms, const size_t& t_level)
{
    return dnlpost_alpha<f_digamma, 1>(f_digamma(), t_alpha, t_ms, t_level) * t_alpha + 1.f;
}

float
Ops::ddlpost_lalpha(const float& t_alpha, MultiScaleMap& t_ms, const size_t& t_level)
{
    return dnlpost_alpha<f_trigamma, 1>(f_trigamma(), t_alpha, t_ms, t_level) * t_alpha + dnlpost_alpha<f_trigamma, 2>(f_trigamma(), t_alpha, t_ms, t_level) * t_alpha * t_alpha;
}

float
Ops::lpost_lalpha(float& t_alpha, MultiScaleMap& t_ms, const size_t& t_level)
{
    return dnlpost_alpha<f_lgamma, 0>(f_lgamma(), t_alpha, t_ms, t_level) + log(t_alpha);
}

void
Ops::update_scale_model(scalemodelType& t_scl_mdl, ExpMap& t_expmap, CountsMap& t_bkg_map)
{
    const auto npixels = t_expmap.get_npixels();
    if (TempDS::m_Fltv.count("update_scale_model") == 0) {
        TempDS::m_Fltv["update_scale_model"] = AllocateAligned<float>(npixels);
    }
    tagF a;
    vecF total_exp_v = Set(a, 0.0);
    Utils::mul_store_add_pixels<vecF, tagF>(npixels, Constants::get_max_lanes(), t_expmap.get_map(), t_bkg_map.get_img_map(), TempDS::m_Fltv["update_scale_model"], total_exp_v);
    float total_exp = GetLane(SumOfLanes(total_exp_v));

    float total_cnt = Utils::reduce<tagF, float>(npixels, t_bkg_map.get_data_map());

    t_scl_mdl.scale = rgamma(total_cnt + t_scl_mdl.scale_pr, 1 / (total_exp + t_scl_mdl.scale_exp));
}

AsyncParamFileIO::AsyncParamFileIO(std::string t_out_file, const ExpMap& t_exp_map, const MultiScaleMap& t_ms, const Config& t_conf)
  : AsyncFileIO(t_out_file)
{
    //print the header
    m_out_file << m_get_cmnt_str() << "Code will run in posterior sampling mode.\n";
    if (t_conf.is_fit_bkg_scl())
        m_out_file << m_get_cmnt_str() << "A scale parameter will be fit to the bkg model.\n";
    m_out_file << m_get_cmnt_str() << "The total number of Gibbs draws is " << t_conf.get_max_iter() << "\n";
    m_out_file << m_get_cmnt_str() << "Every " << t_conf.get_save_thin() << "th draw will be saved\n";
    m_out_file << m_get_cmnt_str() << "The model will be fit using the Multi Scale Prior.\n";
    m_out_file << m_get_cmnt_str() << "The data matrix is " << t_exp_map.get_dim() << " by " << t_exp_map.get_dim() << "\n";
    m_out_file << m_get_cmnt_str() << "The data file should contain a  2^" << t_ms.get_nlevels() << " by 2^" << t_ms.get_nlevels() << " matrix of counts.\n";
    m_out_file << "Starting Values for the smoothing parameter (alpha):\n";
    for (size_t i = 0; i < t_ms.get_nlevels() - 1; ++i) {
        m_out_file << m_get_cmnt_str() << "Aggregation level: " << i << ",     alpha: " << t_ms.get_alpha(i)
                   << (i == 0 ? "  (Full data)" : i == t_ms.get_nlevels() - 1 ? "  (2x2 table)"
                                                                              : "")
                   << "\n";
    }
    m_out_file << m_get_cmnt_str() << "The prior distribution on the total count from the multiscale component is\n";
    m_out_file << m_get_cmnt_str() << "Gamma(" << t_ms.get_ttlcnt_pr() << ", " << t_ms.get_ttlcnt_exp() << ")\n";
    m_out_file << m_get_cmnt_str() << "The hyper-prior smoothing parameter (kappa 2) is " << t_ms.get_al_kap2() << ".\n";

    //print the table header
    *this << "Iteration"
          << "logPost"
          << "stepSize"
          << "cycleSpinRow"
          << "cycleSpinCol";
    for (size_t i = 0; i < t_ms.get_nlevels(); ++i) {
        *this << "smoothingParam" << i;
    }
    *this << "expectedMSCounts"
          << "bkgScale";
}

void
AsyncImgIO::write_img(CountsMap& t_map)
{
    auto dim = t_map.get_dim();
    auto npixels = t_map.get_npixels();
    const auto& img = t_map.get_img_map();
    for (size_t i = 0; i < npixels; i += dim) {
        for (size_t j = 0; j < npixels; ++j) {
            m_out_file << img.get()[i + j] << " ";
        }
        m_out_file << "\n";
    }
}

void
image_analysis_R(
  float* t_outmap,
  float* t_post_mean,
  float* t_cnt_vector,
  float* t_src_vector,
  float* t_psf_vector,
  float* t_map_vector,
  float* t_bkg_vector,
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
  float* t_alpha_init,
  int* t_alpha_init_len,
  float* t_ms_ttlcnt_pr,
  float* t_ms_ttlcnt_exp,
  float* t_ms_al_kap2,
  float* t_ms_al_kap1,
  float* t_ms_al_kap3)
{
    Config conf(*t_max_iter, *t_burn, *t_save_iters, *t_save_thin, *t_fit_bkg_scl);
    /* The psf */
    PSF psf_map(*t_nrow_psf, *t_nrow_psf, t_psf_vector, "psf_map");
    /* The observed counts and model */
    CountsMap obs_map(*t_nrow, *t_ncol, CountsMapDataType::DATA, "observed_map", t_cnt_vector);
    /* The source counts and model (starting values) */
    CountsMap src_map(*t_nrow, *t_ncol, CountsMapDataType::IMG, "source_map", t_src_vector);
    /* The background counts and model */
    CountsMap bkg_map(*t_nrow, *t_ncol, CountsMapDataType::IMG, "background_map", t_bkg_vector);
    /* The deblurred counts and model */
    CountsMap deblur_map(*t_nrow, *t_ncol, CountsMapDataType::BOTH, "deblur_map");
    /* The exposure map */
    ExpMap exp_map(*t_nrow, *t_ncol, t_map_vector, "exposure_map");

    /* The multiscale map */
    MultiScaleMap ms_map(obs_map.get_power2(), t_alpha_init, *t_ms_al_kap1, *t_ms_al_kap2, *t_ms_al_kap3, *t_ms_ttlcnt_pr, *t_ms_ttlcnt_exp);

    /* Re-Norm bkg->img into same units as src->img [AVC Oct 2009] */
    bkg_map.re_norm_img(exp_map.get_max_exp());

    scalemodelType bkg_scale(1.0 /*starting value*/, 0.001 /*prior value*/, 0.0 /*prior value*/);

    llikeType llike;

    AsyncImgIO out_img_file(*t_out_filename);
    AsyncParamFileIO out_param_file(*t_param_filename, exp_map, ms_map, conf);

    bayes_image_analysis(t_outmap, t_post_mean, out_img_file, out_param_file, conf, psf_map, exp_map, obs_map, deblur_map, src_map, bkg_map, ms_map, llike, bkg_scale);
}

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
  llikeType& t_llike,
  scalemodelType& t_bkg_scale)
{
    set_seed(123, 456);
    size_t npixels = t_obs.get_npixels();
    /* Initlaize the R Random seed */
    //GetRNGState();

    //compute_expmap() => because wrapping is the only behaviour pr=map

    Ops::update_deblur_image(t_expmap, t_deblur, t_src, t_bkg, t_bkg_scale);

    t_llike.cur = Ops::comp_ms_prior(t_src, t_ms);

    /************************************/
    /************MAIN LOOP***************/
    /************************************/

    for (t_conf.iter = 1; t_conf.iter <= t_conf.get_max_iter(); t_conf.iter++) {
        if (t_conf.is_save()) {
            t_param_file << "\n"
                         << t_conf.iter;
        }
    }

    /********************************************************************/
    /*********    REDISTRIBUTE obs.data to deblur.data            *******/
    /*********    SEPERATE deblur.data into src.data and bkg.data *******/
    /*********    IF NECESSARY add  cnts to src.data              *******/
    /********************************************************************/

    Ops::redistribute_counts(t_psf, t_deblur, t_obs, t_llike);
    Ops::remove_bkg_from_data(t_deblur, t_src, t_bkg, t_bkg_scale);
    Ops::add_cnts_2_adjust_4_exposure(t_expmap, t_src);

    //no monotone convergence checks -> MCMC
    if (t_conf.is_save()) {
        // Rprintf("Current Log-Posterior: %10g", llike->cur);
        t_param_file << t_llike.cur;
        if (t_conf.iter > 1) {
            t_param_file << t_llike.cur - t_llike.pre;
        } else {
            t_param_file << 0.0;
        }
    }
    t_llike.pre = t_llike.cur;

    t_llike.cur = Ops::update_image_ms(t_param_file, t_expmap, t_src, t_ms, t_conf);

    /***************************************************************/
    /*********          UPDATE BACKGROUND MODEL            *********/
    /***************************************************************/

    if (t_conf.is_fit_bkg_scl()) {
        /* ExpMap Added Per Jason Kramer 13 Mar 2009 */

        Ops::update_scale_model(t_bkg_scale, t_expmap, t_bkg);
        if (t_conf.is_save()) {
            t_param_file << t_bkg_scale.scale;
        }
    } /* if fit background scale */

    Ops::update_deblur_image(t_expmap, t_deblur, t_src, t_bkg, t_bkg_scale);

    if (t_conf.is_save()) {
        t_out_file.write_img(t_src);
    }
    if (t_conf.is_save_post_mean()) {
        float m = t_conf.get_iter_m_burn();
        const auto& src_img = t_src.get_img_map();
        for (size_t i = 0; i < npixels; ++i) {
            t_post_mean[i] = ((m - 1) * t_post_mean[i] + src_img.get()[i]) / m;
        }
    }

    //PutRNGState();
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
} // namespace HWY_NAMESPACE
} // namespace hwy
HWY_AFTER_NAMESPACE();

//g++ -c lira.cpp -lR -lhwy -I/usr/local/include/ -I/usr/share/R/include/ -O2 -std=c++17 -ltbb
//g++ -shared lira.cpp -lR -lhwy -I/usr/local/include/ -I/usr/share/R/include/ -O2 -std=c++17 -ltbb -fPIC -o lira.so