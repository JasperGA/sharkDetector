#ifndef SHARKFIND_HPP
#define SHARKFIND_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SSIM.hpp"
#include <iostream>


using namespace std;
using namespace cv;

Mat src, dst, morph, gray, thresh;
char window_name[] = "Smoothing Demo";
RNG rng(12345);

int erosion_size = 5;
int dilation_size = 3;
int morph_size = 10;

Rect sharkFind( Mat & img_src )
{
    //namedWindow( window_name, WINDOW_AUTOSIZE );

    //src = imread( "shark_4.jpeg", IMREAD_COLOR );

    //  medianBlur is looking like a good option, SEE below
    medianBlur( img_src, dst, 5);

    // Opening morph to get filter out light reflections
    Mat element = getStructuringElement(  MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx( dst, morph, MORPH_OPEN, element );
    //imshow("opening", morph);
    //waitKey();

    // Creating a threshold rather then edge detection as a test
    //medianBlur( morph, morph, 5);
    GaussianBlur( morph, morph, Size( 5, 5 ), 0, 0 );
    
    // Convert from BGR to HSV colorspace
    cvtColor(morph, gray, COLOR_BGR2HSV);

    Rect mostBound;
    double maxSSIM = 0;
    double curSSIM;
    // Detect the object based on HSV Range Values
    for (int low_V = 100; low_V <= 120; low_V += 5) {
        inRange(gray, Scalar(0, 0, low_V), Scalar(255, 255, 255), thresh);

        //waitKey();
        //threshold( gray, thresh, 100, 255, THRESH_BINARY);
        //imshow("thresh", thresh);
        //waitKey();

        // Getting contours
        Mat canny_output;
        Canny( thresh, canny_output, 100, 100*2 );
        
        vector<vector<Point> > contours;
        findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
        
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );

        vector<Rect> realRect( contours.size() );
        vector<Moments> realMu( contours.size() );

        vector<Moments> mu( contours.size() );

        for( size_t i = 0; i < contours.size(); i++ )
        {
            approxPolyDP( contours[i], contours_poly[i], 3, true );
            boundRect[i] = boundingRect( contours_poly[i] );
            mu[i] = moments( contours[i] );
        }

        Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        int x = 0;
        for( size_t i = 0; i< contours.size(); i++ )
        {
            // If cotour is large enough, draw it and add to list of contourrs to check
            if ((boundRect[i].width * boundRect[i].height) > 10000) {
                Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
                drawContours( drawing, contours_poly, (int)i, color );
                rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );
                // add list to check contours
                realRect[x] = boundRect[i];
                realMu[x] = mu[i];
                x++;
            }
        }
        
        //imshow( "Contours", drawing );

        // Template matching
        Mat img_display, result, templ, templ_down;
        thresh.copyTo( img_display );

        // add loop here to go resize templ to each contour and check image
        double minVal; double maxVal; Point minLoc; Point maxLoc;
        Point matchLoc;
        Point conCentre;

        for( int i = 0; i< x; i++ )
        {
            matchLoc = realRect[i].tl();
            Rect myROI(matchLoc.x, matchLoc.y, realRect[i].width, realRect[i].height);
            Mat roi = thresh(myROI);
            conCentre.x = realMu[i].m10 / realMu[i].m00;
            conCentre.y = realMu[i].m01 / realMu[i].m00; 
            if (realRect[i].width > realRect[i].height) {
                if (conCentre.x > (matchLoc.x + realRect[i].width/2)) {
                    templ = imread("shark_out.jpeg" , IMREAD_GRAYSCALE);
                } else {
                    templ = imread("shark_out_left.jpeg" , IMREAD_GRAYSCALE);
                }
            } else {
                if (conCentre.y > (matchLoc.y + realRect[i].height/2)) {
                    templ = imread("shark_out_down.jpeg" , IMREAD_GRAYSCALE);
                } else {
                    templ = imread("shark_out_up.jpeg" , IMREAD_GRAYSCALE);
                }
            }
            resize(templ, templ, Size(realRect[i].width, realRect[i].height));
            threshold( templ, templ, 200, 255, 0 );
            
            curSSIM = qm::compute_quality_metrics(roi, templ, 4);
            if (curSSIM > maxSSIM) {
                maxSSIM = curSSIM;
                mostBound = realRect[i];
                /*
                imshow( "roi", roi);
                imshow( "temp", templ);
                waitKey();
                */
            }
        }
    }
    return mostBound;
}

#endif