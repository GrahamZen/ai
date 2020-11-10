#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <map>
#include <set>
#include <iostream>
#include <algorithm>
using namespace std;

enum emotion
{
    anger, disgust, fear, joy, sad, surprise, unknown
};
const string emoStr[] = {
    "anger", "disgust", "fear", "joy", "sad", "surprise"
};

class sentence
{
private:
    emotion e;
    int wordCnt = 0;
    void getEmotion(const string &s) {
        if (s == "anger") {
            e = anger;
        }
        else if (s == "disgust") {
            e = disgust;
        }
        else if (s == "fear") {
            e = fear;
        }
        else if (s == "joy") {
            e = joy;
        }
        else if (s == "sad") {
            e = sad;
        }
        else if (s == "surprise") {
            e = surprise;
        }
        else if (s == "?") {
            e = unknown;
        }
        else throw invalid_argument((string("Emotion ")+s+string(" do not exist!")).c_str());//
    }
    map<string, int> termFreq;
    set<string>word;
public:
    sentence() = default;
    explicit sentence(const string &s) {
        load(s);
    }
    emotion getEmotion()const {
        return e;
    }
    void load(const string &s) {
        string tmpStr;
        stringstream ss(s);
        auto emoPos = s.find_last_of(',');
        string emo = s.substr(emoPos+1);
        getEmotion(emo);
        ss.str(s.substr(0, emoPos));
        while (ss >> tmpStr) {
            wordCnt++;
            word.insert(tmpStr);
            termFreq[tmpStr]++;
        }
    }
    const set<string>& getWordSet() const{
        return word;
    }
    float dist(const sentence&sent) {
        auto w = word;
        const auto& w2 = sent.word;
        const auto& w1 = word;
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
        return sqrt(Dist);
    }
    ~sentence() = default;
    const map<string, int>& getTF()const {
        return termFreq;
    }
};

class dataSet
{
private:
    multiset<sentence> data;
    //IDF
    map<string, float> InvDocFreq;
    //how many sentence the word appeared in
    vector<string> word;
    vector<int> wordFreq;
    vector<vector<float>> TF_IDF;

    void TF_IDF_init() {
        if (sentenceCnt == 0 || wordCnt == 0)
            throw invalid_argument("Data hasn't been initialized!");
        TF_IDF = vector<vector<float>>(sentenceCnt, vector<float>(wordFreq.size(), 0));
        int i = 0;
        string currTerm;
        for (auto iter = data.begin(); iter != data.end(); ++iter, ++i)
        {
            for (int j = 0; j < wordCnt; j++)
            {
                currTerm = word[j];
                const auto& TF = iter->getTF();
                if (TF.find(word[j]) != TF.end())
                    TF_IDF[i].at(j) = TF.at(currTerm) * InvDocFreq[currTerm];
                else TF_IDF[i].at(j) = 0;
            }
        }
    }
    void data_init(ifstream& infile) {
        map<string, int> wordFreqMap;
        string s;
        while (getline(infile, s))
        {
            sentence sent;
            sent.load(s);
            data.insert(sent);
            auto TF = sent.getTF();
            for (auto& t : TF) {
                wordFreqMap[t.first]++;
            }
        }
        word.resize(wordFreqMap.size());
        wordFreq.resize(wordFreqMap.size());
        transform(wordFreqMap.begin(), wordFreqMap.end(), word.begin(), [](pair<string, int> p) { return p.first; });
        transform(wordFreqMap.begin(), wordFreqMap.end(), wordFreq.begin(), [](pair<string, int> p) { return p.second; });
    }
public:
    dataSet() = default;
    dataSet(ifstream& infile) {
        load(infile);
    }
    void load(ifstream& infile) {
        data_init(infile);
        sentenceCnt = data.size();
        wordCnt = wordFreq.size();
        for (int i = 0; i < wordCnt; i++) {
            InvDocFreq[word[i]] = log((float)sentenceCnt / (1+wordFreq[i]));
        }
        TF_IDF_init();
    }
    const vector<vector<float>>& getTF_IDF()const {
        return TF_IDF;
    }
    int sentenceCnt = 0;
    int wordCnt = 0;
    ~dataSet() = default;
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
    emotion classify(const sentence &target) {
        vector<pair<emotion,float>> vec(size);
        for (int i = 0; i < size; i++)
        {
            vec[i].first = data->at(i).getEmotion();
            vec[i].second = data->at(i).dist(target);
        }
        sort(vec.begin(), vec.end(), [](const pair<emotion, float>&p1,const pair<emotion, float>&p2) {return p1.second < p2.second; });
        map<emotion,int>kElem;
        for (int i = 0; i < k; i++)
        {
            kElem[vec[i].first]++;
        }
        auto res = std::max_element
        (
            kElem.begin(),kElem.end() ,
            [](const pair<emotion,int>& p1, const pair<emotion, int>& p2) {
                return p1.second < p2.second;
            }
        );
        return res->first;
    }

private:
    int k;
    vector<sentence>* data;
    size_t size;
};


int main(int argc, char const* argv[])
{
    string s;
    ifstream train_set("lab1_data\\classification_dataset\\train_set.csv");
    KNN k(train_set);
    train_set.close();
    int bestK = 0;float bestCR = 0;
    for (int i = 1; i < 20; i++)
    {
        ifstream validation_set("lab1_data\\classification_dataset\\validation_set.csv");
        k.setK(i);
        int correct = 0, total = 0;
        getline(validation_set, s);
        while (getline(validation_set, s)) {
            auto sent = sentence(s);
            if (sent.getEmotion() == k.classify(sent)) {
                correct++;
            }
            total++;
        }
        cout <<"K = "<<i<< ",correct rate=" << (float)correct / total << endl;
        validation_set.close();
        if (bestCR < (float)correct / total) {
            bestK = i,bestCR= (float)correct / total;
        }
    }
    cout <<"best K = "<<bestK<< ",best correct rate=" << bestCR<< endl;
    ifstream test_set("lab1_data\\classification_dataset\\test_set.csv");
    ofstream outfile("17338233_zhenggehan_KNN_classification.csv");
    k.setK(bestK);
    getline(test_set, s);
    while (getline(test_set, s)) {
        s = s.substr(s.find(',') + 1);
        auto sent = sentence(s);
        outfile << emoStr[k.classify(sent)] << endl;
    }
    test_set.close();
    return 0;
}
