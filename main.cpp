#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <bitset>
#include <fstream>
using namespace cv;
using namespace std;

#define CLASSES 24
int SHIFT_COUNT = 0;
#define PROBLEMS "/problems.txt"

uchar find_min(uchar & min)
{
	for(int i = 1; i < 8; ++i)
		if(min > (uchar) ((min << i) | (min >> (8 - i))))
			min = (uchar) ((min << i) | (min >> (8 - i)));
	return min;
}

uchar eight_n(const Mat & input, int y, int x)
{
	uchar val = input.at<uchar>(y, x) >> SHIFT_COUNT;
	uchar min = 255;
	for(int i = -1; i < 2; ++i)
		for(int j = -1; j < 2; ++j)
			if(!(i == 0 && 0 == j))
			{
				uchar b = input.at<uchar>(y + i, x + j) >> SHIFT_COUNT;
				min |= val > b ? 1 : 0;
				min <<= 1;
			}
	return find_min(min);
}

void calculate_lbp(const Mat & input, vector<float> & histogram)
{
	Mat dst;
	//equalizeHist(input, dst);
	input.copyTo(dst);
	//GaussianBlur(input, dst, Size(), 3, 3);
	for(int y = 1; y < dst.rows - 1; ++y)
		for(int x = 1; x < dst.cols - 1; ++x)
			++histogram[eight_n(dst, y, x)];
	
}

int test_calculation(const vector<pair<vector<float>, int>> & all, const vector<float> & the_histogram, int type, int & result)
{
	vector<pair<int, float>> distances;
	for (int i = 0; i < all.size(); ++i)
	{
		float data = 0;
		for (int each_value = 0; each_value < the_histogram.size(); ++ each_value)
			data += pow(all[i].first[each_value] - the_histogram[each_value], 2.0) ;//* (each_value + 1);
		//unnecessary square root op is getting ignoerd.
		//data = sqrt(data);
		distances.push_back(pair<int, float>(all[i].second, data));
	}
	sort(distances.begin(), distances.end(),[](const pair<int, float>& a, const pair<int, float>& b) -> bool
		{return  b.second > a.second;});
	
	//cout << "------------------------" << endl;
	//std::cout << "error on: " << type << ". guess: " << distances[0].first << '\n';
	result = distances.front().first;
	int temp = 0;
	for (auto &&a : distances)
	{
		//cout << "Distance to " << a.second << " type " << a.first << endl;
		if(a.first == type)
			break;
		++temp;	
	}
	//std::cout << "\tGap: " << temp << std::endl;
	return temp;
}

void open_read_calc(const string & path, vector <float> & histogram)
{
	Mat src = imread( path, IMREAD_GRAYSCALE );
	if( src.empty() )
	{
		cout << "Could not open or find the image at %" << path << "%\n";
		exit(EXIT_FAILURE);
	}
	calculate_lbp(src, histogram);
}

void test(const string & base_path, const string & test_case, vector<pair<vector<float>, int>> & all_vector)
{
	ifstream images_txt(base_path + "/" + test_case + "/test.txt");
	int nm;
	images_txt >> nm;
	for (int a = 0; a < nm; ++a)
	{
		int type;
		string file_name;
		images_txt >> file_name >> type;
		vector<float> feature_local(255, 0);
		file_name = base_path + "/images/" + file_name.substr(0, file_name.find('.')) + ".png";
		open_read_calc(file_name, feature_local);
		all_vector.push_back(pair(feature_local, type));
	}
	images_txt.close();
}

void train(const string & base_path, const string & test_case, const vector<pair<vector<float>, int>> & all)
{
	int nm;
	float overall_miss = 0,
			rate = 0;
	ifstream images_txt(base_path + "/" +  test_case + "/train.txt");
	images_txt >> nm;
	for (int a = 0; a < nm; ++a)
	{
		int type, guess;
		string file_name;
		vector<float> the_histogram(255, 0);
		
		images_txt >> file_name >> type;
		file_name = base_path + "/images/" + 
					file_name.substr(0, file_name.find('.')) + ".png";
		
		open_read_calc(file_name, the_histogram);
		int temp;
		if((temp = test_calculation(all, the_histogram, type, guess)) == 0) ++rate;
		overall_miss += temp;
		cout << file_name << " = guess: " << guess << ", actual: " << type << endl;
	}
	images_txt.close();
	std::cout << "rate for " << test_case << "= " << 100.0 * rate / (float) nm 
			  << ". Overall miss gap: " << overall_miss / (float) (nm - rate) << std::endl;
}


int main( int argc, char** argv )
{
	int nm;
	string path(argv[1]);
	ifstream cases(path + PROBLEMS);
	cases >> nm;
	for(int i = 0; i < nm; ++i)
	{
		string test_c;
		cases >> test_c;
		for(SHIFT_COUNT = 0; SHIFT_COUNT < 7; ++SHIFT_COUNT)
		{
			cout << "For the dataset " << test_c << ", after ignoring " << SHIFT_COUNT << " lsb of features" << endl;
			vector<pair<vector<float>, int>> all;
			test(path, test_c, all);
			train(path, test_c, all);
		}
	}
	cases.close();
    return 0;
}
