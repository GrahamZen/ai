#pragma once
#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
using namespace std;
const int type_of_emotion = 6;
enum class emotion :int
{
    anger, disgust, fear, joy, sad, surprise, unknown
};
enum class codeType :int
{
    one_hot, tf_idf
};
const string emoStr[] = {
    "anger", "disgust", "fear", "joy", "sad", "surprise"
};
const string codeTypeStr[] = {
    "one-hot","tf-idf"
};
