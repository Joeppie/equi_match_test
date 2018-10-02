#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


int main()
{

  Mat equi1 = imread("/home/joep/Philip_VO_code_minimal/demo_data/5D4FMDM5.jpg");
  Mat equi2 = imread("/home/joep/Philip_VO_code_minimal/demo_data/5D4FMDMG.jpg");

  assert(equi1.size().width == 4800);
  assert(equi2.size().width == 4800);

  std::cout <<  equi1.size().width << " x " << equi1.size().height << std::endl;

  const auto matcher = cv::Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(cv::NORM_L2));
  cv::Ptr<AgastFeatureDetector> detector = AgastFeatureDetector::create(8);




  Mat mask1 = Mat::zeros(equi1.size(), CV_8U);  // type of mask is CV_8U
  auto rect = cv::Rect(2133.0,183.0,2283.0-2133.0,279.0-183.0);

  rect.x -= 100;
  rect.width += 100;
  rect.y -= 100;
  rect.height += 100;

  Mat roi1(mask1, rect);   // roi is a sub-image of mask specified by cv::Rect object
  roi1 = Scalar(255);   // we set elements in roi region of the mask to 255

  Mat mask2 = Mat::zeros(equi2.size(), CV_8U);  // type of mask is CV_8U
  auto rect2 = cv::Rect(2934.0,441.0,3069.0-2934.0,528.0-441.0);

  rect2.x -= 100;
  rect2.width += 100;
  rect2.y -= 100;
  rect2.height += 100;

  Mat roi2(mask2, rect2);   // roi is a sub-image of mask specified by cv::Rect object

  roi2 = Scalar(255);   // we set elements in roi region of the mask to 255

  equi1 = equi1(rect);
  equi2 = equi2(rect2);


  //Resize image to be larger; for better visualization; but matching takes place on this
/*
  resize(equi1,equi1,{equi1.cols*3,equi1.rows*3},0,0,INTER_CUBIC);
  resize(equi2,equi2,{equi2.cols*3,equi2.rows*3},0,0,INTER_CUBIC);

*/


  //Code for matching entire equirectangular, but restricting matching to region of interested
  vector<KeyPoint> kp1,kp2;
  //detect
  detector->detect(equi1,kp1/*,mask1*/);
  detector->detect(equi2,kp2/*,mask2*/);

  cv::Mat descriptors1;
  cv::Mat descriptors2;

  //compute
  cv::Ptr<DescriptorExtractor> descriptor = DAISY::create(40);
  descriptor->compute(
      equi1,
      kp1,
      descriptors1);

  descriptor->compute(
      equi2,
      kp2,
      descriptors2);

  //knn-2 match, no ratio test.
  vector<vector<DMatch>> matches;
  vector<vector<DMatch>> ratio_tested_matches;
  matcher->knnMatch(descriptors1,descriptors2,matches,2);

  for(auto results: matches)
  {
    const auto& best   = results[0];
    const auto& second = results[1];
    //consider ratio-test passed if second is 25% worse in relative distance between matches than second.
    if(best.distance * 1.05 < second.distance)
    {
      ratio_tested_matches.emplace_back(results);
    }
  }

  //draw.
  Mat result;
  drawMatches(
      equi1,kp1,
      equi2,kp2,
      ratio_tested_matches, //drawing ratio-tested matches only.
      result, cv::Scalar(0, 255, 0),cv::Scalar::all(-1),std::vector<std::vector<char> >(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

 // Mat resized = Mat::zeros(1200,4800,result.type());

 // resize(result,resized,{result.cols/2,result.rows/2},0,0,INTER_LANCZOS4);
  imshow("equi matches test", result);

  waitKey(100000);

  return 0;
}