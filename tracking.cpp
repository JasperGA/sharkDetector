#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/tracking.hpp>
#include <opencv4/opencv2/core/ocl.hpp>

#include "sharkSSIMFind.hpp"

using namespace cv;
using namespace std;

// Convert to string for displaying FPS
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.4.1
    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};
    // vector <string> trackerTypes(types, std::end(types));

    // Create a tracker
    string trackerType = trackerTypes[2];

    Ptr<Tracker> tracker;

    if (trackerType == "BOOSTING")
        tracker = TrackerBoosting::create();
    if (trackerType == "MIL")
        tracker = TrackerMIL::create();
    if (trackerType == "KCF")
        tracker = TrackerKCF::create();
    if (trackerType == "TLD") // good
        tracker = TrackerTLD::create();
    if (trackerType == "MEDIANFLOW")
        tracker = TrackerMedianFlow::create();
    if (trackerType == "GOTURN")
        tracker = TrackerGOTURN::create();
    if (trackerType == "MOSSE") // good
        tracker = TrackerMOSSE::create();
    if (trackerType == "CSRT") // good
        tracker = TrackerCSRT::create();

    // Read video
    VideoCapture video("sharkV_2.mp4");

    // Following is for saving video produced
    int frame_width = video.get(CAP_PROP_FRAME_WIDTH); 
    int frame_height = video.get(CAP_PROP_FRAME_HEIGHT);
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G'); 
    VideoWriter recor("KCF_Tracker.avi", codec, 20, Size(frame_width,frame_height));
    
    // Exit if video is not opened
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl; 
        return 1; 
    } 

    // Read first frame 
    Mat frame; 
    bool ok = video.read(frame); 

    // Define initial bounding box 
    Rect2d bbox = sharkFind(frame); 

    // Uncomment the line below to select a different bounding box 
    // bbox = selectROI(frame, false); 
    // Display bounding box. 
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 ); 

    imshow("Tracking", frame); 
    tracker->init(frame, bbox);
    
    int i = 1;
    while(video.read(frame))
    {     
        // Start timer
        double timer = (double)getTickCount();
        
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
        
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
        
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
            //waitKey();
        } else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
            //bbox = sharkFind(frame);
        }
        
        // Experimenting with trying to correct drift and occulsion errors
        /*
        if (i % 10 == 0) 
        {
            bbox = sharkFind(frame);
        }
        */

        // Display tracker type on frame
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Display and save frame.
        imshow("Tracking", frame);
        recor.write(frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
        i++;
    }
}
