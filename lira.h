#ifndef LIRA_H
#define LIRA_H

#include "hwy/foreach_target.h"

#include <stddef.h>
#include <stdio.h>

#include <memory>
#include <numeric>

#include <iostream>
#include <memory>
#include <hwy/base.h>
#include <string>
#include "hwy/aligned_allocator.h"
#include "hwy/highway.h"
#include "hwy/nanobenchmark.h"
#include <R.h>
#include <math.h>
#include <Rmath.h>
#include <exception>
#include <sstream>
#include <algorithm>
#include <map>
#include <numeric>
#include <execution>
typedef std::stringstream sstr;

namespace hwy
{
    namespace HWY_NAMESPACE
    {
        typedef AlignedFreeUniquePtr<float[]> uPtr_F;
        const auto &execParUnseq = std::execution::par_unseq;
        struct llikeType
        {
            double cur; /* the current log likelihood of the iteration */
            double pre; /* the previous log likelihood of the iteration */
        };

        class Ops
        {
        public:
            //all the operations assume that the length input vectors are a factor of Lanes(d). May fault if not.
            template <typename ArithOp>
            inline static void v_op(ArithOp t_op, size_t t_npixels, const uPtr_F &t_a, const uPtr_F &t_b, uPtr_F &t_result)
            {
                const ScalableTag<float> a;
                const ScalableTag<float> b;
                for (size_t i = 0; i < t_npixels; i += Constants::max_lanes())
                {
                    const auto a_vec = Load(a, t_a.get() + i);
                    const auto b_bec = Load(b, t_b.get() + i);
                    Store(t_op(Load(a, t_a.get() + i), Load(b, t_b.get() + i)), Constants::d, t_result.get() + i);
                }
            }

            inline static void redistribute_counts(const PSF &t_psf, CountsMap &t_deblur, const CountsMap t_obs, llikeType &t_llike)
            {
                t_deblur.set_data_zero();
            }
        };
        class Constants
        {
        public:
            inline static const int MultiplyDeBruijnBitPosition2[32]{
                0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
                31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9};
            inline static const std::map<MapType, std::string> map_names = {{MapType::PSF, "PSF"}, {MapType::EXP, "Exposure"}, {MapType::COUNTS, "Counts"}};
            inline static const size_t get_lanes()
            {
                ScalableTag<float> d;
                return static_cast<size_t>(Lanes(d));
            }
            inline static const ScalableTag<float> d;
            inline static const size_t max_lanes() { return Lanes(d); };
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

        class IncorrectDimensions : virtual public std::exception
        {
        protected:
            std::string m_err_msg;

        public:
            explicit IncorrectDimensions(const std::string &t_msg) : m_err_msg(t_msg){};

            virtual const char *what() const throw()
            {
                return m_err_msg.c_str();
            }
        };

        template <typename T>
        class ImgMap
        {
        public:
            ImgMap(size_t t_nrows, size_t t_ncols, MapType t_map_type) : m_nrows(t_nrows), m_ncols(t_ncols), map_type(t_map_type), map_name(Constants::map_names(t_map_type)), n_pixels(m_nrows * m_ncols)
            {
                check_sizes();
                static_cast<T *>(this).initialize();

                if (map_type != MapType::PSF)
                {
                    //http://www.graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
                    max_levels = Constants::MultiplyDeBruijnBitPosition2[static_cast<uint32_t>(m_nrows * 0x077CB531U) >> 27];
                }
            }

            size_t get_num_pixels()
            {
                return n_pixels;
            }

        protected:
            const size_t m_nrows;
            const size_t m_ncols;
            size_t max_levels{1};
            MapType map_type;
            std::string map_name;
            size_t n_pixels;

        private:
            void check_sizes()
            {
                if (m_nrows != m_ncols)
                {
                    sstr ss;
                    ss << 'Incorrect ' << map_name << ' map dimensions: ' << map_name << ' map must be a square.\n';
                    ss << ' Input dimensions nrows: ' << m_nrows << ', ncols: ' << m_ncols;
                    throw(IncorrectDimensions(ss.str()));
                }

                if (map_type == MapType::PSF && m_nrows % 2 == 0)
                {
                    std::cout << 'Warning: The PSF must have an odd dimension to allow the maximum to be exactly at the center of the image';
                }

                if (map_type != MapType::PSF && (m_nrows == 0 || (m_nrows & (m_nrows - 1)) || m_nrows < 8))
                {
                    sstr ss;
                    ss << "Incorrect " << map_name << " map dimensions: Each side must be a power of 2 and greater than 8.\n";
                    ss << "Input dimensions: nrows: " << m_nrows << " ncols: " << m_ncols;
                    throw(IncorrectDimensions(ss.str()));
                }
            }
        };

        class WrappedIndexMap
        {
        public:
            WrappedIndexMap(size_t &t_img_dim, size_t &t_psf_dim) : m_img_dim(t_img_dim), m_psf_dim(t_psf_dim)
            {
                m_pad_size = m_psf_dim / 2; //assumes an odd sized PSF with the maximum at the center
                m_pad_img_dim = m_pad_size + m_img_dim;
                m_npixels = pow(2, m_pad_img_dim);

                m_idx_map = AllocateAligned<size_t>(m_npixels);
                size_t _a, _b;

                for (size_t i = 0; i < m_pad_img_dim; ++i)
                {
                    for (size_t j = 0; j < m_pad_img_dim; ++j)
                    {

                        m_idx_map.get()[i * m_pad_img_dim + j] = wrap_idx(i - m_pad_size, m_pad_img_dim) * m_pad_img_dim + wrap_idx(j - m_pad_size, m_pad_img_dim);
                    }
                }
            }

        protected:
            size_t m_img_dim;
            size_t m_psf_dim;
            size_t m_pad_img_dim;
            size_t m_pad_size; //Padding on each side of the image. Elements in the padded region will be wrapped
            size_t m_npixels;  //total pixels in the map inclusive of the padding
            AlignedFreeUniquePtr<size_t[]> m_idx_map{nullptr};

        private:
            size_t wrap_idx(const size_t &t_idx, const size_t &t_dim)
            {
                if (t_idx >= 0)
                    return t_idx;

                while (t_idx < 0)
                    t_idx += 1;
                return t_idx % t_dim;
            }
        };
        class PSF : ImgMap<PSF>
        {
            PSF(size_t t_nrows, size_t t_ncols, const float *t_psf_mat) : ImgMap<PSF>{t_nrows, t_ncols, MapType::PSF}, m_mat_holder(t_psf_mat)
            {
            }

            PSF &initialize()
            {
                m_orig_size = m_nrows * m_ncols;
                ScalableTag<float> d;
                m_aligned_size = m_orig_size + m_orig_size % Lanes(d); //pad the extra elements with zeros

                m_mat = AllocateAligned<float>(m_aligned_size);
                m_inv = AllocateAligned<float>(m_aligned_size);

                std::copy(execParUnseq, m_mat_holder, m_mat_holder + m_orig_size, m_mat.get());
                std::copy(execParUnseq, m_mat_holder, m_mat_holder + m_orig_size, m_rmat.get());
                std::reverse(execParUnseq, m_rmat.get(), m_rmat.get() + m_orig_size);
                std::fill(execParUnseq, m_mat.get() + m_orig_size, m_mat.get() + m_aligned_size, 0.f);

                //TODO--compute L,R,U,D
            }

        protected:
            uPtr_F m_mat;
            uPtr_F m_inv;
            uPtr_F m_rmat; //180 deg CCW rotated matrix, a.k.a reversed
            size_t m_L{0}, m_R{0}, m_U{0}, m_D{0};
            size_t m_orig_size;
            size_t m_aligned_size;

        private:
            const float *m_mat_holder; //temporary storage before initialization checks
        };

        class ExpMap : ImgMap<ExpMap>
        {
            ExpMap(size_t t_nrows, size_t t_ncols, const float *t_expmap) : m_mat_holder(t_expmap), ImgMap<ExpMap>(t_nrows, t_ncols, MapType::EXP) {}

            void initialize()
            {
                m_map = AllocateAligned<float>(n_pixels);
                //pr_det = AllocateAligned<float>(n_pixels);
                prod = AllocateAligned<float>(n_pixels);

                std::copy(m_mat_holder, m_mat_holder + n_pixels, m_map.get());
                max_exp = *std::max_element(m_map.get(), m_map.get() + n_pixels);

                ScalableTag<float> d;
                const auto max_vector = Set(d, max_exp);

                //normalize the exposure map
                for (size_t i = 0; i < n_pixels; i += Lanes(d))
                {
                    const auto a = Load(d, m_map.get() + i);
                    Store(Div(a, max_vector), d, m_map.get() + i);
                }

                //Because wrapping is the enforced default behaviour => pr_det=1, hence prod=m_map
                std::copy(m_map.get(), m_map.get() + n_pixels, prod.get());
            }

        protected:
            uPtr_F m_map;
            uPtr_F pr_det;
            uPtr_F prod;

        private:
            const float *m_mat_holder;
            size_t n_pixels{0};
            float max_exp{1.f};
            size_t max_levels{1};
        };

        class CountsMap : ImgMap<CountsMap>
        {
        public:
            CountsMap(size_t t_nrows, size_t t_ncols, float *t_map, CountsMapDataType t_map_data_type) : ImgMap(t_nrows, t_ncols, MapType::COUNTS), m_map_holder(t_map), map_data_type(t_map_data_type) {}

            CountsMap &initialize()
            {
                data = AllocateAligned<float>(n_pixels);
                img = AllocateAligned<float>(n_pixels);

                if (map_data_type == CountsMapDataType::DATA)
                {
                    std::copy(m_map_holder, m_map_holder + n_pixels, data.get());
                    set_img_zero();
                }
                else
                {
                    std::copy(m_map_holder, m_map_holder + n_pixels, img.get());
                    set_data_zero();
                }
            }

            void set_data_zero()
            {
                std::fill(data.get(), data.get() + n_pixels, 0.f);
            }

            void set_img_zero()
            {
                std::fill(img.get(), img.get() + n_pixels, 0.f);
            }

        protected:
            uPtr_F data;                     //the counts
            uPtr_F img;                      //the image (expected counts)
            CountsMapDataType map_data_type; //type of the data that should be read in

        private:
            const float *m_map_holder;
        };

        class MultiScaleLevelMap
        {
        public:
            MultiScaleLevelMap(size_t t_dimension, float t_alpha) : m_dimension(t_dimension), m_dimension_agg(m_dimension / 2), m_npixels(pow(t_dimension, 2)), m_npixels_agg(m_npixels / 4), m_current_map(AllocateAligned<float>(m_npixels)), m_agg_norm_map(AllocateAligned<float>(m_npixels)), m_agg_indices(m_npixels_agg, 0), m_curr_map_agg_indices(m_npixels, 0), m_curr_map_indices(m_npixels, 0), m_alpha(t_alpha)
            {
                /*, m_row_interleaved_sum(AllocateAligned<float>(m_half_npixels)), m_row_interleaved_A(AllocateAligned<float>(m_half_npixels)), m_row_interleaved_B(AllocateAligned<float>(m_npixels_agg)), m_col_interleaved_A(AllocateAligned<float>(m_npixels_agg)), m_row_to_col_interleaved_flag(m_half_npixels, TRUE), m_row_to_col_indices(m_half_npixels, 0), m_row_interleaved_indices(m_half_npixels, 0), m_sub_npixels(m_npixels / 4), m_half_npixels(m_npixels / 2)*/

                std::fill(m_current_map.get(), m_current_map.get() + m_npixels, 0.f);
                std::fill(m_agg_norm_map.get(), m_agg_norm_map.get() + m_npixels, 0.f);
                /*std::fill(m_row_interleaved_sum.get(), m_row_interleaved_sum.get() + m_half_npixels, 0.f);
                std::fill(m_row_interleaved_A.get(), m_row_interleaved_A.get() + m_half_npixels, 0.f);
                std::fill(m_row_interleaved_B.get(), m_row_interleaved_B.get() + m_half_npixels, 0.f);
                std::fill(m_col_interleaved_A.get(), m_col_interleaved_A.get() + m_npixels_agg, 0.f);
                std::fill(m_col_interleaved_B.get(), m_col_interleaved_B.get() + m_npixels_agg, 0.f);*/

                //pre-compute slicing indices
                // size_t counter = 0;
                // std::for_each(m_row_to_col_interleaved_flag.begin(), m_row_to_col_interleaved_flag.end(), [](auto &i)
                //               {
                //                   i = counter % 2 == 0 ? TRUE : FALSE;
                //                   ++counter;
                //               });
                // std::for_each(m_row_to_col_indices.begin(), m_row_to_col_indices.end(), [](auto &i)
                //               { i = i % 2 == 0 ? i : i - 1; });
                // std::iota(m_row_interleaved_indices.begin(), m_row_interleaved_indices.end(), 0);
                std::iota(m_agg_indices.begin(), m_agg_indices.end(), 0);
                std::iota(m_curr_map_indices.begin(), m_curr_map_indices.end(), 0);
                std::for_each(m_agg_indices.begin(), m_agg_indices.end(), [&](const size_t &i)
                              {
                                  size_t main = 0;
                                  if (i < m_dimension_agg)
                                  {
                                      main = i * 2;
                                  }
                                  else
                                  {

                                      main = i / m_dimension_agg * m_dimension_agg * 4 + (i - i / m_dimension_agg * m_dimension_agg) * 2;
                                  }
                                  m_curr_map_agg_indices[main] = i;
                                  m_curr_map_agg_indices[main + 1] = i;
                                  m_curr_map_agg_indices[main + m_dimension] = i;
                                  m_curr_map_agg_indices[main + m_dimension + 1] = i;
                              });

                for (auto i = 0; i < 4; i++)
                {
                    m_curr_sub_maps.push_back(AllocateAligned<float>(m_npixels_agg));
                }
                for (auto i = 0; i < m_dimension * (m_dimension - 2); i += m_dimension)
                {
                    for (auto j = 0; j < m_dimension_agg; j++)
                    {
                        m_curr_sub_maps_idx[0].push_back((i + j) * 2);
                        m_curr_sub_maps_idx[1].push_back((i + j) * 2 + 1);
                        m_curr_sub_maps_idx[2].push_back((i + j) * 2 + m_dimension);
                        m_curr_sub_maps_idx[3].push_back((i + j) * 2 + m_dimension + 1);
                    }
                }
            }

            void set_map(const uPtr_F &t_data_map)
            {
                std::copy(t_data_map.get(), t_data_map.get() + m_npixels, m_current_map.get());
                set_sub_maps();

                //set the row interleaved levels
                //set_row_interleaved();
            }

            void set_sub_maps()
            {
                for (auto &i : m_agg_indices)
                {
                    m_curr_sub_maps[0].get()[i] = m_current_map.get()[m_curr_sub_maps_idx[0][i]];
                    m_curr_sub_maps[1].get()[i] = m_current_map.get()[m_curr_sub_maps_idx[1][i]];
                    m_curr_sub_maps[2].get()[i] = m_current_map.get()[m_curr_sub_maps_idx[2][i]];
                    m_curr_sub_maps[3].get()[i] = m_current_map.get()[m_curr_sub_maps_idx[3][i]];
                }
            }

            uPtr_F &get_map()
            {
                return m_current_map;
            }

            void get_aggregate(uPtr_F &t_out_data_map, bool t_norm = TRUE)
            { //set the aggregate to t_data_map. Contains m_npixels/4 pixels

                if (m_npixels_agg / 4 >= Constants::max_lanes())
                {
                    //compute on lanes
                    add_4(m_npixels_agg, m_curr_sub_maps[0], m_curr_sub_maps[1], m_curr_sub_maps[2], m_curr_sub_maps[3], t_out_data_map);
                }
                else
                {
                    //else use the good old for loop
                    for (auto i = 0; i < m_npixels_agg; ++i)
                    {
                        t_out_data_map.get()[i] = std::reduce(m_curr_sub_maps.begin(), m_curr_sub_maps.end(), 0.f, [&](auto a, auto b)
                                                              { return a.get()[i] + b.get()[i]; });
                    }
                }
                if (t_norm)
                    normalize_curr_map(t_out_data_map);
            }

            float get_level_log_prior()
            {
                float log_prior = std::transform_reduce(
                    m_current_map.get(), m_current_map.get() + m_npixels, 0.f,
                    [](const auto &i)
                    { return log(i); },
                    std::plus<float>());

                log_prior *= m_alpha - 1;
            }

        protected:
            // uPtr_F m_row_interleaved_A;   //even rows of the current map
            // uPtr_F m_row_interleaved_B;   //odd rows of the current map
            // uPtr_F m_col_interleaved_A;   //even colums of the row interleaved sum
            // uPtr_F m_col_interleaved_B;   //odd rows of the row interleaved sum
            uPtr_F m_current_map;  //image of the current level
            uPtr_F m_agg_norm_map; //
            //uPtr_F m_row_interleaved_sum; // the sum of even and odd rows. Has a size of~ dim/2 x dim
            /* Divide the current map into 4 sub maps. Their sum would be the aggregate matrix. Only do it if m_npixels/4 > max_lanes */
            std::vector<uPtr_F> m_curr_sub_maps;
            std::vector<std::vector<size_t>> m_curr_sub_maps_idx;
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
            std::vector<size_t> m_agg_indices;          //0-to-m_npixels_agg index array
            std::vector<size_t> m_curr_map_agg_indices; //each element specifies its correspoinding index in the agg map
            std::vector<size_t> m_curr_map_indices;     //0-to-m_npixels index array
            float m_alpha;

        private:
            void add_4(size_t t_npixels, const uPtr_F &t_a, const uPtr_F &t_b, const uPtr_F &t_c, const uPtr_F &t_d, uPtr_F &t_out)
            {
                const ScalableTag<float> a, b, c, d;
                const auto lanes = Constants::max_lanes();
                for (auto i = 0; i < t_npixels; i += lanes)
                {
                    Store(
                        (Load(a, t_a.get() + i) + Load(b, t_b.get() + i) + Load(c, t_c.get() + i) + Load(d, t_d.get() + i)), Constants::d, t_out.get() + i);
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

            void generate_agg_norm_map(const uPtr_F &t_out_data_map)
            {

                std::for_each(m_curr_map_indices.begin(), m_curr_map_indices.end(), [&](const auto &i)
                              { m_agg_norm_map.get()[i] = t_out_data_map.get()[(m_curr_map_agg_indices[i]]; });
            }

            void normalize_curr_map(uPtr_F &t_out_data_map)
            {
                generate_agg_norm_map(t_out_data_map);
                Ops::v_op(Div<decltype(Constants::d)>, m_npixels, m_current_map, m_agg_norm_map, m_current_map);
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
            MultiScaleMap(CountsMap &t_counts_map)
            {
            }

        protected:
            //std::vector
        };

    }
}

#endif