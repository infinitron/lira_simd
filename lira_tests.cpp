

#include "lira.cpp"
#include "gtest/gtest.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {


using T = AlignedFreeUniquePtr<double[]>;
using Tv = AlignedFreeUniquePtr<double*[]>; //the "view" array
using vecF = Vec<ScalableTag<double>>;
using tagF = ScalableTag<double>;

class MapsTest : public ::testing::Test
{
    //create an image map with all ones
    //the psf is just a delta function
  protected:
    void SetUp() override
    {
        //allocate the memory
        img_npixels = img_dim * img_dim;
        psf_npixels = psf_dim * psf_dim;
        image_ones = new double[img_npixels];
        bkg_zero = new double[img_npixels];
        psf_seq = new double[psf_npixels];
        psf_reverse = new double[psf_npixels];

        //set the test pixels
        std::fill(image_ones, image_ones + img_npixels, 1.0);
        std::fill(bkg_zero, bkg_zero + img_npixels, 1.0);
        std::fill(psf_seq, psf_seq + psf_npixels, 0.0);
        psf_seq[psf_dim * psf_dim / 2] = 1.0;
        psf_map = new PSF<T, tagF>(psf_dim, psf_dim, psf_seq, "psf");
        std::reverse_copy(psf_seq, psf_seq + psf_npixels, psf_reverse);

        immap = new CountsMap<T, Tv>(img_dim, img_dim, CountsMapDataType::DATA, "imdata", image_ones);
        srcmap = new CountsMap<T, Tv>(img_dim, img_dim, CountsMapDataType::IMG, "img", image_ones);
        bkgmap = new CountsMap<T, Tv>(img_dim, img_dim, CountsMapDataType::IMG, "img", image_ones);

        deblur_m = new CountsMap<T, Tv>(img_dim, img_dim, CountsMapDataType::BOTH, "deblur");
        exp_m = new ExpMap<T, tagF>(img_dim, img_dim, image_ones, "expmap");

        bkg_scale = new scalemodelType(1.0 /*starting value*/, 0.001 /*prior value*/, 0.0 /*prior value*/);

        //perform ops
        ms_map = new MultiScaleMap<T, Tv, tagF>(5, alpha_vals, 0, 1000, 3, 1, 0.05);
        ms_map->set_init_level_map(*srcmap);
        //ms_map->compute_cascade_agregations(TRUE);

        Ops::update_deblur_image<vecF>(*exp_m, *deblur_m, *srcmap, *bkgmap, *bkg_scale);
        llk.pre = 0;
        llk.cur = 0;
    }

    double* image_ones; //image with all ones--template for obs,map,start
    double* bkg_zero;   //image with zero bkg
    double* psf_seq;    //psf with a sequence of numbers
    double* psf_reverse;
    CountsMap<T, Tv>* immap{ nullptr };
    CountsMap<T, Tv>* bkgmap{ nullptr };
    CountsMap<T, Tv>* srcmap{ nullptr };
    CountsMap<T, Tv>* deblur_m{ nullptr };
    ExpMap<T, tagF>* exp_m{ nullptr };
    MultiScaleMap<T, Tv, tagF>* ms_map{ nullptr };
    PSF<T, tagF>* psf_map{ nullptr };
    size_t img_dim{ 32 };
    size_t psf_dim{ 15 };
    double alpha_vals[5]{ 0.5, 0.5, 0.5, 0.5, 0.5 };
    void CntsMap_is_set();
    int img_npixels;
    int psf_npixels;
    scalemodelType* bkg_scale;
    llikeType llk;
};

TEST_F(MapsTest, CntsMap_is_set)
{
    const auto& map = immap->get_data_map();
    EXPECT_TRUE(std::equal(image_ones, image_ones + img_npixels, map.get()));
}

TEST_F(MapsTest, PSF_is_set)
{
    psf_map = new PSF<T, tagF>(psf_dim, psf_dim, psf_seq, "psf");
    const auto& rmat = psf_map->get_rmat();
    EXPECT_TRUE(std::equal(psf_reverse, psf_reverse + psf_npixels, rmat.get()));
}


TEST_F(MapsTest, deblur_update_test)
{
    const auto& deblur_img = deblur_m->get_img_map();
    auto npixels = img_dim * img_dim;
    double val = 2;
    ASSERT_TRUE(std::all_of(deblur_img.get(), deblur_img.get() + npixels, [&](auto a) { return a == val; }));
}

TEST_F(MapsTest, ms_prior_comp)
{
    double sum = 0;
    auto nlevels = ms_map->get_nlevels() - 1;
    double dim = img_dim;
    for (auto i = 0; i < nlevels; ++i) {
        sum += (alpha_vals[i] - 1) * dim * dim * log(0.25); //all the pixels become 0.25
        dim /= 2;
    }
    sum += (ms_map->get_ttlcnt_pr() - 1) * img_dim * img_dim - (ms_map->get_ttlcnt_exp() * img_dim * img_dim);
    EXPECT_NEAR(sum, Ops::comp_ms_prior(*srcmap, *ms_map), 1e-6);

    EXPECT_NEAR(img_dim * img_dim * log(0.25) * (alpha_vals[0] - 1), ms_map->get_level(0).get_level_log_prior(), 1e-6);
}

TEST_F(MapsTest, ms_levels_check)
{
    auto nlevels = ms_map->get_nlevels();
    EXPECT_EQ(nlevels, 6);
    //the top level starts with  a value of 1
    double val = 1.0 / 4.; //pixel value in each level after normalization
    ms_map->compute_cascade_agregations(true);
    for (auto i = 0; i < nlevels - 1; i++) {

        const auto level_npixels = ms_map->get_level(i).get_dim() * ms_map->get_level(i).get_dim();

        const auto& level_map = ms_map->get_level(i).get_map();

        ASSERT_TRUE(std::all_of(level_map.get(), level_map.get() + level_npixels, [&](auto a) { return a == val; })) << "Iteration: " << ::testing::PrintToString(i);
    }
    ASSERT_DOUBLE_EQ(ms_map->get_max_agg_value(), img_dim * img_dim);
}

TEST_F(MapsTest, redist_counts_check)
{
    //if all the pixels are set to 1 for src and bkg, delblur.img would have each pixel set to 2
    //after convolving it with a delta function and renormalizing it each pixel in deblur.data should 
    //set to 1
    //This would also test the consistency of the input wraped matrix

    //take a delta function
    std::fill(psf_seq, psf_seq + psf_npixels, 0.0);
    
    

    WrappedIndexMap wmap(deblur_m->get_dim(), psf_map->get_dim());
    deblur_m->set_wraped_mat(wmap, CountsMapDataType::IMG);
    deblur_m->set_wraped_mat(wmap, CountsMapDataType::DATA);
    Ops::redistribute_counts<vecF, T, Tv, tagF>(*psf_map, *deblur_m, *immap, llk);

    //the deblur data and deblur img must be equal
    const auto& dat = deblur_m->get_data_map();
    const auto& img = deblur_m->get_img_map();
    
    ASSERT_TRUE(std::equal(dat.get(), dat.get() + img_dim * img_dim, img.get(),[&](auto a,auto b) { return a == b/2; }));
}

TEST_F(MapsTest,spin_row_test){

    //set spin row=2 and spin col=0
    const auto &data_orig=srcmap->get_data_map();
    auto temp_v=AllocateAligned<double>(img_npixels);

    srcmap->get_spin_data(temp_v,2,0);

    //the second to last row in the original image must be equal to the second row in the result vector
    ASSERT_TRUE(std::equal(data_orig.get()+img_dim*(img_dim-3),data_orig.get()+img_dim*(img_dim-2),temp_v.get()+img_dim,temp_v.get()+2*img_dim));

}

TEST_F(MapsTest,spin_col_test){

    //set spin row=0 and spin col=2
    const auto &data_orig=srcmap->get_data_map();
    auto temp_v=AllocateAligned<double>(img_npixels);

    srcmap->get_spin_data(temp_v,0,2);

    //the first dim-2 elements in any row in the original must equal dim-2 elemnts on the result with an offset of 2
    //compare the first row
    EXPECT_TRUE(std::equal(data_orig.get(),data_orig.get()+img_dim-2,temp_v.get()+img_dim+1));

    //the last column in the original must be equal to the second column in the result
    EXPECT_DOUBLE_EQ(*(data_orig.get()+img_dim-1),*(temp_v.get()+1));
}

TEST_F(MapsTest,log_priors_test){

    auto nlevels = ms_map->get_nlevels();
    for(auto i=0;i<nlevels-1;i++){
        auto prior=ms_map->get_log_prior(0.5,0.);
        EXPECT_DOUBLE_EQ(prior,-1000*pow(0.5,3));
    }
    for(auto i=0;i<nlevels-1;i++){
        auto prior=ms_map->get_log_prior(0.5,1);
        EXPECT_DOUBLE_EQ(prior,-3000*pow(0.5,2));
    }
    for(auto i=0;i<nlevels-1;i++){
        auto prior=ms_map->get_log_prior(0.5,2);
        EXPECT_DOUBLE_EQ(prior,-6000*pow(0.5,1));
    }
}
TEST_F(MapsTest,lpost_lalpha){
    //level 0 contains all 0.25
    ms_map->compute_cascade_agregations(true);
    auto val=ms_map->get_level(0).get_map()[0];
    EXPECT_EQ(0.25,val);

    //level 0 logpost
    set_seed(123,456);
    double suma=0;
    auto dim0=ms_map->get_level(0).get_dim();
    for(auto i=0;i<dim0;i++){
        for(auto j=0;j<dim0;++j){
            suma+=lgammafn(0.25/*val*/ + 0.5/*alpha*/);
        }
    }
    set_seed(123,456);
    EXPECT_DOUBLE_EQ(suma,ms_map->get_level(0).get_ngamma_sum<f_lgamma>(f_lgamma(),0.5));

    //level 1
    EXPECT_EQ(0.25,ms_map->get_level(1).get_map()[0]);
    double sumb=0;
    auto dim1=ms_map->get_level(1).get_dim();
     for(auto i=0;i<dim1;i++){
        for(auto j=0;j<dim1;++j){
            sumb+=lgammafn(0.25/*val*/ + 4.*0.5/*alpha*/);
        }
    }

    set_seed(123,456);
    EXPECT_NEAR(sumb,ms_map->get_level(1).get_ngamma_sum<f_lgamma>(f_lgamma(),4*.5),2);

    
    size_t level=0;
    set_seed(123,456);
    double alpha=0.5;
    double sum=-1000*pow(0.5,3)+suma-sumb+dim1*dim1*(lgammafn(4*0.5)-4*lgammafn(0.5))+log(alpha);
    auto val1=Ops::lpost_lalpha(alpha,*ms_map,level);
    EXPECT_NEAR(val1,sum,1);
}

TEST_F(MapsTest,dlpost_lalpha){

    double factor = 4;
    //level 0 contains all 0.25
    ms_map->compute_cascade_agregations(true);
    auto val=ms_map->get_level(0).get_map()[0];
    EXPECT_EQ(0.25,val);

    //level 0 logpost
    set_seed(123,456);
    double suma=0;
    auto dim0=ms_map->get_level(0).get_dim();
    for(auto i=0;i<dim0;i++){
        for(auto j=0;j<dim0;++j){
            suma+=digamma(0.25/*val*/ + 0.5/*alpha*/);
        }
    }
    set_seed(123,456);
    EXPECT_DOUBLE_EQ(suma,ms_map->get_level(0).get_ngamma_sum<f_digamma>(f_digamma(),0.5));

    //level 1
    EXPECT_EQ(0.25,ms_map->get_level(1).get_map()[0]);
    double sumb=0;
    auto dim1=ms_map->get_level(1).get_dim();
     for(auto i=0;i<dim1;i++){
        for(auto j=0;j<dim1;++j){
            sumb+= digamma(0.25/*val*/ + 4.*0.5/*alpha*/);
        }
    }

    set_seed(123,456);
    EXPECT_NEAR(sumb,ms_map->get_level(1).get_ngamma_sum<f_digamma>(f_digamma(),4*.5),2);

    double sum=-3000*pow(0.5,2)+suma-factor*sumb+dim1*dim1*(factor * digamma(4*0.5)-4*digamma(0.5));
    size_t level=0;
    set_seed(123,456);
    double alpha=0.5;
    auto val1=Ops::dlpost_lalpha(alpha,*ms_map,level);
    EXPECT_NEAR(val1,sum*alpha+1,2);
}

TEST_F(MapsTest,ddlpost_lalpha){

    double factor = 16;
    //level 0 contains all 0.25
    ms_map->compute_cascade_agregations(true);
    auto val=ms_map->get_level(0).get_map()[0];
    EXPECT_EQ(0.25,val);

    //level 0 logpost
    set_seed(123,456);
    double suma=0;
    auto dim0=ms_map->get_level(0).get_dim();
    for(auto i=0;i<dim0;i++){
        for(auto j=0;j<dim0;++j){
            suma+=trigamma(0.25/*val*/ + 0.5/*alpha*/);
        }
    }
    set_seed(123,456);
    EXPECT_DOUBLE_EQ(suma,ms_map->get_level(0).get_ngamma_sum<f_trigamma>(f_trigamma(),0.5));

    //level 1
    EXPECT_EQ(0.25,ms_map->get_level(1).get_map()[0]);
    double sumb=0;
    auto dim1=ms_map->get_level(1).get_dim();
     for(auto i=0;i<dim1;i++){
        for(auto j=0;j<dim1;++j){
            sumb+= trigamma(0.25/*val*/ + 4.*0.5/*alpha*/);
        }
    }

    set_seed(123,456);
    EXPECT_NEAR(sumb,ms_map->get_level(1).get_ngamma_sum<f_trigamma>(f_trigamma(),4*.5),2);

    auto log_prior=-6000*pow(0.5,1);
    EXPECT_DOUBLE_EQ(log_prior,ms_map->get_log_prior(0.5,2));

    double sum=log_prior+suma-factor*sumb+dim1*dim1*(factor * trigamma(4*0.5)-4.*trigamma(0.5));

    size_t level=0;
    set_seed(123,456);
    double alpha=0.5;
    auto ddlpost_alpha=Ops::dnlpost_alpha<f_trigamma, 2>(f_trigamma(), alpha, *ms_map, level);
    EXPECT_DOUBLE_EQ(sum,ddlpost_alpha);
    sum *=alpha*alpha;
    sum+=(Ops::dlpost_lalpha(alpha,*ms_map,level)-1)*alpha;
    set_seed(123,456);
    auto val1=Ops::ddlpost_lalpha(alpha,*ms_map,level);
    EXPECT_NEAR(val1,sum,2);
}
 }
}
HWY_AFTER_NAMESPACE();


//compile command for the test
//g++ lira_tests.cpp -lR -lhwy -lgtest -lgtest_main -I/usr/local/include/ -I/usr/share/R/include/ -O2 -std=c++17 -ltbb  -o lira_tests.exe -lRmath -fPIC -lpthread;./lira_tests.exe
