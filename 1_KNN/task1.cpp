#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
using namespace std;


class sentence
{
private:
    int id;
    int wordCnt = 0;
    string str;
    void load(const string& s) {
        string tmpStr;
        stringstream ss(str);
        while (ss >> tmpStr) {
            wordCnt++;
            word.insert(tmpStr);
            termFreq[tmpStr]++;
        }
        for (auto& str : termFreq) {
            str.second /= wordCnt;
        }
    }
    string getStr(const string& target) {
        stringstream ss(target);
        ss >> id;
        auto tabPos = target.find_last_of('\t');
        return target.substr(tabPos + 1);
    }
    map<string, float> termFreq;
    set<string>word;
public:
    sentence() = default;
    explicit sentence(const string& s) :str(getStr(s)) {
        load(s);
    }
    ~sentence() = default;
    string getStr()const {
        return str;
    }
    float getWordFreq(string s)const {
        return termFreq.at(s);
    }
    const set<string>& getWordSet() const {
        return word;
    }
    const map < string, float >& getTF()const {
        return termFreq;
    }
    friend bool operator<(const sentence& s1, const sentence& s2) { return s1.id < s2.id; }
};

class dataSet
{
private:
    vector<sentence> data;
    //IDF
    vector < float > InvDocFreq;
    //how many sentence the word appeared in
    vector<string> word;
    vector<int> wordFreq;
    vector<vector<float>> TF_IDF;
    bool TF_IDF_inited = false;
    void data_init(ifstream& infile) {
        int line_no = 0;
        set<sentence>sentenceSet;
        map<string, int> wordFreqMap;
        string s;
        while (getline(infile, s))
        {
            sentence sent(s);
            sentenceSet.insert(sent);
            auto TF = sent.getTF();
            for (auto& t : TF) {
                wordFreqMap[t.first]++;
            }
            sentenceCnt++;
        }
        wordCnt = wordFreqMap.size();
        word.resize(wordCnt);
        wordFreq.resize(wordCnt);
        data.resize(sentenceCnt);
        transform(wordFreqMap.begin(), wordFreqMap.end(), word.begin(), [](pair<string, int> p) { return p.first; });
        transform(wordFreqMap.begin(), wordFreqMap.end(), wordFreq.begin(), [](pair<string, int> p) { return p.second; });
        transform(sentenceSet.begin(), sentenceSet.end(), data.begin(), [](sentence s) { return s; });
        //此后data有序
    }
    void TF_IDF_init() {
        if (sentenceCnt == 0 || wordCnt == 0)
            throw invalid_argument("Data hasn't been initialized!");
        TF_IDF = vector<vector<float>>(sentenceCnt, vector<float>(wordCnt, 0));
        int i = 0;
        string currTerm;
        for (int i = 0; i < sentenceCnt; i++)
        {
            for (int j = 0; j < wordCnt; j++)
            {
                currTerm = word[j];
                const auto& TF = data[i].getTF();
                if (TF.find(currTerm) != TF.end())
                    TF_IDF.at(i).at(j) = TF.at(currTerm) * InvDocFreq[j];
                else TF_IDF.at(i).at(j) = 0;
            }
        }
        TF_IDF_inited = true;
    }
public:
    dataSet() = default;
    dataSet(ifstream& infile) {
        load(infile);
    }
    void load(ifstream& infile) {
        data_init(infile);
        InvDocFreq.resize(wordCnt);
        for (int i = 0; i < wordCnt; i++) {
            InvDocFreq[i] = log10((float)sentenceCnt / (1 + wordFreq[i]));
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



int main(int argc, char const* argv[])
{
    string s;
    ifstream infile("lab1_data\\semeval.txt");
    dataSet d(infile);
    const auto& mat = d.getTF_IDF();
    ofstream outfile("semeval_output.csv");
    for (int i = 0; i < d.sentenceCnt; i++) {
        for (int j = 0; j < d.wordCnt; j++)
        {
            outfile << mat[i].at(j) << ',';
        }
        outfile << endl;
    }

    return 0;
}
