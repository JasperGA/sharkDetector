// Works with openCV 2.4
/*
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
*/
// works with openCV 4.2
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

#include "SSIM.hpp"

using namespace std;
using namespace cv;

Mat src, dst, morph, gray, thresh;
char window_name[] = "SSIM Video";

int erosion_size = 5;
int dilation_size = 3;
int morph_size = 10;

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main( int argc, char ** argv )
{
    Mat up, down, left, right;
    up = imread("shark_out_up.jpeg" , IMREAD_GRAYSCALE);
    threshold( up, up, 200, 255, 0 );

    down = imread("shark_out_down.jpeg" , IMREAD_GRAYSCALE);
    threshold( down, down, 200, 255, 0 );
    
    right = imread("shark_out.jpeg" , IMREAD_GRAYSCALE);
    threshold( right, right, 200, 255, 0 );
    
    left = imread("shark_out_left.jpeg" , IMREAD_GRAYSCALE);
    threshold( left, left, 200, 255, 0 );


    Point prev = Point(0, 0);

    namedWindow( window_name, WINDOW_AUTOSIZE );
    VideoCapture cap("sharkV_2.mp4");
    
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH); 
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G'); 
    VideoWriter video("SharkSSIM.avi", codec, 20, Size(frame_width,frame_height));

    while (true) {
        cap >> src;
        if(src.empty())
        {
            break;
        }

        // Initialise timer to get frame rate
        double timer = (double)getTickCount();

        //  medianBlur is looking like a good option, SEE below
        medianBlur( src, dst, 5);

        // Opening morph to get filter out light reflections
        Mat element = getStructuringElement(  MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
        morphologyEx( dst, morph, MORPH_OPEN, element );
        //imshow("opening", morph);
        //waitKey();

        // Creating a threshold rather then edge detection as a test
        //medianBlur( morph, morph, 5);
        //GaussianBlur( morph, morph, Size( 5, 5 ), 0, 0 );
        
        // Convert from BGR to HSV colorspace
        cvtColor(morph, gray, COLOR_BGR2HSV);


        // Main for loop to test SSIM on different HSV thresholds 
        Rect mostBound;
        double maxSSIM = 0;
        double curSSIM;
        // Detect the object based on HSV Range Values
        for (int low_V = 100; low_V <= 110; low_V += 10) {
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
            
            // Vectors for contour polygons and bounding rectangles of polygons
            vector<vector<Point> > contours_poly( contours.size() );
            vector<Rect> boundRect( contours.size() );

            // Vectors for sustible bounding rectangles and polygon moments
            vector<Rect> realRect( contours.size() );
            vector<Moments> realMu( contours.size() );
            vector<Moments> mu( contours.size() );

            // For each contour, get bounding rectangle and moment from its polygon 
            for( size_t i = 0; i < contours.size(); i++ )
            {
                approxPolyDP( contours[i], contours_poly[i], 3, true );
                boundRect[i] = boundingRect( contours_poly[i] );
                mu[i] = moments( contours[i] );
            }


            // For each bounding rectangle, get important contours and count how many 
            int x = 0;
            for( size_t i = 0; i< contours.size(); i++ )
            {
                // If contour is right size and in region where last ROI was found (or ROI doesn't exist, prerv = (0,0)),
                // add its bounding box and moment to list of contours to check with SSIM
                if ((boundRect[i].x < (prev.x + 200) && boundRect[i].x > (prev.x - 200)) || prev.x == 0) {
                    if ((boundRect[i].y < (prev.y + 50) && boundRect[i].y > (prev.y - 50)) || prev.y == 0) {
                        if ((boundRect[i].width * boundRect[i].height) > 2000 && (boundRect[i].width * boundRect[i].height) < 50000) {
                            realRect[x] = boundRect[i];
                            realMu[x] = mu[i];
                            x++;
                        }
                    }
                }
            }
            
            // If no corect contours were found, set previous ROI to point (0,0) 
            if (x == 0) {
                prev = Point(0, 0);
            }

            Mat img_display, result, templ, templ_down;
            double minVal; double maxVal; Point minLoc; Point maxLoc;
            Point matchLoc;
            Point conCentre;

            // For each contour of interest, find most likely base image that matches and calculate SSIM
            for( int i = 0; i< x; i++ )
            {
                // Extract ROI for frame "thresh"
                matchLoc = realRect[i].tl();
                Rect myROI(matchLoc.x, matchLoc.y, realRect[i].width, realRect[i].height);
                Mat roi = thresh(myROI);

                // Find centre of contour
                conCentre.x = realMu[i].m10 / realMu[i].m00;
                conCentre.y = realMu[i].m01 / realMu[i].m00;

                /*
                    Assume sharks are longer then they are wide and that their head is wider then their tail.
                    If ROI width > height, check if center of contour is to the left or right, if left use shark facing 
                    left as base image vis versa for right. Same up/down image as well.
                */
                if (realRect[i].width > realRect[i].height) {
                    if (conCentre.x > (matchLoc.x + realRect[i].width/2)) {
                        templ = right;
                    } else {
                        templ = left;
                    }
                } else {
                    if (conCentre.y > (matchLoc.y + realRect[i].height/2)) {
                        templ = down;
                    } else {
                        templ = up;
                    }
                }

                // Resize base image to match size of ROI
                resize(templ, templ, Size(realRect[i].width, realRect[i].height));
                
                // Calculate SSIM between contour and base image, if SSIM is large then max previously found
                // set bounding box to current ROI
                curSSIM = qm::compute_quality_metrics(roi, templ, 4);
                if (curSSIM > maxSSIM) {
                    maxSSIM = curSSIM;
                    mostBound = realRect[i];
                    prev.x = mostBound.x;
                    prev.y = mostBound.y;
                }
        
                //imshow( "roi", roi);
                //imshow( "temp", templ);
                //waitKey();
            }
        }

        // Draw bounding box around what is thought to be shark
        Scalar color = Scalar( 0, 0, 255 );
        rectangle( src, mostBound.tl(), mostBound.br(), color, 3 );

        // Calclate FPS and display on video
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
        putText(src, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        
        imshow(window_name, src);
        video.write(src);

        char key = (char) waitKey(30);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }
    
    return 0;
}