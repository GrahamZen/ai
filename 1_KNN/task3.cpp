#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <set>
#include <array>
#include <algorithm>
#include <numeric>
using namespace std;
const int type_of_emotion=6;



enum class emotion: int
{
    anger=0, disgust, fear, joy, sad, surprise, unknown
};
const string emoStr[type_of_emotion] = {
    "anger", "disgust", "fear", "joy", "sad", "surprise",
};
class sentence
{
private:
    array<float, type_of_emotion>emoPercent;
    int wordCnt = 0;
    map<string, int> termFreq;
    set<string>words;
    template<size_t N>
    float COR(const array<float,N>&X,const array<float,N>&Y) {
        float sum = accumulate(X.begin(), X.end(), 0.0);
        float Xmean = sum / N;
        sum = accumulate(Y.begin(), Y.end(), 0.0);
        float Ymean = sum / N;
        float cov = 0, squareOfSigmaX = 0, squareOfSigmaY = 0;
        for (int i = 0; i < N; i++)
        {
            cov += (X[i] - Xmean) * (Y[i] - Ymean);
            squareOfSigmaX += pow(X[i] - Xmean, 2);
            squareOfSigmaY += pow(Y[i] - Ymean, 2);
        }
        if (squareOfSigmaX < 0 || squareOfSigmaY < 0)throw overflow_error("line 45:sqrt of minus number!");
        return cov / sqrt(squareOfSigmaX * squareOfSigmaY);
    }

public:
    sentence() = default;
    explicit sentence(const string &s) {
        load(s);
    }
    void load(const string& s) {
        auto pos_of_comma= s.find_first_of(',');
        const auto& sent = s.substr(0, pos_of_comma);
        const auto& dataStr = s.substr(pos_of_comma + 1);
        stringstream ss(sent);
        string str;
        while (ss >> str) {
            wordCnt++;
            words.insert(str);
            termFreq[str]++;
        }
        sscanf_s(dataStr.c_str(), "%f,%f,%f,%f,%f,%f", &emoPercent[0], &emoPercent[1], &emoPercent[2], &emoPercent[3], &emoPercent[4], &emoPercent[5]);
    }
    const set<string>& getWordSet() const{
        return words;
    }
    float dist(const sentence&sent) {
        auto w = words;
        const auto& w2 = sent.words;
        const auto& w1 = words;
        w.insert(w2.begin(), w2.end());
        auto wordSize = w.size();
        float Dist = 0;
        for (const auto&Word:w)
        {
            bool inM1=(w1.find(Word) != w1.end()), inM2=(w2.find(Word) != w2.end());
            int coord1 = inM1 ? termFreq[Word] : 0;
            int coord2 = inM2 ? sent.termFreq.at(Word): 0;
            Dist += (coord1 - coord2) * (coord1 - coord2);
        }
        //if (Dist == 0)throw overflow_error("line 84: distance = 0!");
        return sqrt(Dist);
    }
    ~sentence() = default;
    const map<string, int>& getTF()const {
        return termFreq;
    }
    float get_prob(emotion e) const {
        return emoPercent[(int)e];
    }
    float correlate(const array<float, type_of_emotion>&res) {
        float r= COR<type_of_emotion>(res, emoPercent);
        return r;
    }
};
class KNN
{
public:
    KNN()=default;
    KNN(ifstream &infile):data(new vector<sentence>),k(1) {
        string str;
        getline(infile, str);
        while (getline(infile, str)) {
            sentence sent(str);
            data->push_back(sent);
        }
        size = data->size();
    }
    ~KNN() {
        delete data;
    }
    void setK(int _k) { k = _k; }
    array<float,type_of_emotion> classify(const sentence &target) {
        bool perfect_match=false;
        vector<pair<const sentence*,float>> vec(size);
        array<float, type_of_emotion>res;
        float dist_reverse_sum = 0;
        res.fill(0);
        for (int i = 0; i < size; i++)
        {
            vec[i].second = data->at(i).dist(target);
            vec[i].first = &data->at(i);
        }
        sort(vec.begin(), vec.end(), [](const pair<const sentence*, float>&p1,const pair<const sentence*, float>&p2) {return p1.second < p2.second; });
        perfect_match = (vec[0].second < 1e-4);
        if (perfect_match) {
            int zeroCnt = 0;
            for (int i = 0; i < k; i++)
            {
                if (vec[i].second != 0) continue;
                zeroCnt++;
                for (emotion e = emotion::anger; e < emotion::unknown; e=emotion((int)e+1))
                {
                    res[(int)e] += vec[i].first->get_prob(e);
                }
            }
            transform(res.begin(), res.end(), res.begin(), [zeroCnt](float f) {return f / zeroCnt; });
        }
        else {
            for (int i = 0; i < k; i++)
            {
                if (vec[i].second == 0)throw overflow_error("dist_reverese_sum = inf!");
                dist_reverse_sum += 1 / vec[i].second;
            }
            for (emotion e = emotion::anger; e < emotion::unknown; e=emotion(int(e)+1))
            {
                for (int i = 0; i < k; i++)
                {
                    res[(int)e] += (vec[i].first->get_prob(e)/vec[i].second);
                }
                res[(int)e] /= dist_reverse_sum;
            }
        }
        return res;
    }

private:
    int k;
    vector<sentence>* data;
    size_t size;
};


int main(int argc, char const* argv[])
{
    string s;
    ifstream train_set("lab1_data\\regression_dataset\\train_set.csv");
    KNN k(train_set);
    train_set.close();
    int bestK = 0;float bestCR = 0;
    for (int i = 1; i < 20; i++)
    {
        ifstream validation_set("lab1_data\\regression_dataset\\validation_set.csv");
        k.setK(i);
        float totalCOR= 0, total = 0;
        getline(validation_set, s);
        while (getline(validation_set, s)) {
            auto sent = sentence(s);
            totalCOR += sent.correlate(k.classify(sent));
            total += 1;
        }
        if (total <= 0)throw overflow_error("divided by zero!");
        float corrRate = (float)totalCOR / total;
        cout <<"K = "<<i<< ",correlate rate =" <<corrRate  << endl;
        validation_set.close();
        if (bestCR < corrRate) {
            bestK = i,bestCR= (float)corrRate;
        }
    }
    cout <<"best K = "<<bestK<< ",best correct rate=" << bestCR<< endl;
    ifstream test_set("lab1_data\\regression_dataset\\test_set.csv");
    ofstream outfile("17338233_zhenggehan_KNN_regression.csv");
    k.setK(bestK);
    getline(test_set, s);
    int index = 0;
    outfile << "textid,anger,disgust,fear,joy,sad,surprise" << endl;
    while (getline(test_set, s)) {
        index++;
        auto first_comma=s.find_first_of(',');
        auto last_comma=s.find_last_of(',');
        s = s.substr(first_comma + 1,last_comma-first_comma-1);
        auto sent = sentence(s);
        auto res = k.classify(sent);
        outfile << index
                << ','  << res[(int)emotion::anger] 
                << ','  << res[(int)emotion::disgust] 
                << ','  << res[(int)emotion::fear] 
                << ','  << res[(int)emotion::joy] 
                << ','  << res[(int)emotion::sad] 
                << ','  << res[(int)emotion::surprise] 
                << endl;
    }
    test_set.close();
    return 0;
}
