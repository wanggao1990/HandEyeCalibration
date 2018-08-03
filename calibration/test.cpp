#include "HandEyeCalibration.h"

int main()
{
		//// 标定
		if (true) {
			std::string path = R"(C:\Users\wanggao\Desktop\eyeinhand\图片0731)";
			HandEyeCalibration handEyeTest(path, "out_camera_data.yml", { 7,6 }, 0.025);
			bool ret = handEyeTest.doCalibration();
			if (ret)
			{
				auto camMat = handEyeTest.getCameraMatrix();
				auto distCoffMat = handEyeTest.getDistCoeffsMatrix();
				auto extBigMat = handEyeTest.getExtrinsicsBigMat();
				auto hasCheessVec = handEyeTest.getFoundCheeseBoardVec();
				for (auto& v : hasCheessVec)	
					cout << v << " ";
				cout << endl;
				cout << endl << camMat      << endl;
				cout << endl << distCoffMat << endl;
				cout << endl << extBigMat   << endl << endl;  // {r,t格式}
			}
		}
		
		// X 
		
		std::vector<cv::Mat> vecHg, vecHc;
		bool ret = HandEyeCalibration::readDatasFromFile("out_camera_data.yml", 
			R"(C:\Users\wanggao\Desktop\eyeinhand\图片0731\pose)",	vecHg, vecHc, true);
			
		std::vector<cv::Mat> vecHgij, vecHcij;
		HandEyeCalibration::convertVectors2Hij(vecHg, vecHc, vecHgij, vecHcij);
		cv::Mat Hcg;
		HandEyeCalibration::computerHandEyeMatrix(Hcg, vecHgij, vecHcij, 1);  // Tsai
			
		//////////////////////////////////////////////////////////////////////////
	 	Mat imgPos(1446., 1154. };
		cv::FileStorage fs("HandEyeMatrix.yml", cv::FileStorage::READ);
		Mat X = fs["handEyeMatrix"].mat();
		cv::Point3d pos= HandEyeCalibration::getWorldPos(
			cv::Point2d(Mat{ 1446., 1154. }), z, X, vecHg[0], camMatrix, vecHg[1], vecHc[1]);
		cout << pos << endl;
		
		return 0;
}