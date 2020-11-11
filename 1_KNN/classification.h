#include"typedef.h"

namespace KNN_CLASSIFICATION {
    class sentence
    {
    private:
        emotion e = emotion::unknown;
        int wordCnt = 0;
        string str;
        void load(const string& s) {
            string tmpStr;
            stringstream ss(str);
            while (ss >> tmpStr) {
                wordCnt++;
                word.insert(tmpStr);
                termRelFreq[tmpStr]++;
            }
            for (auto& str : termRelFreq) {
                str.second /= wordCnt;
            }
        }
        string getEmotion(const string& target) {
            auto emoPos = target.find_last_of(',');
            string s = target.substr(emoPos + 1);
            if (s == "anger") {
                e = emotion::anger;
            }
            else if (s == "disgust") {
                e = emotion::disgust;
            }
            else if (s == "fear") {
                e = emotion::fear;
            }
            else if (s == "joy") {
                e = emotion::joy;
            }
            else if (s == "sad") {
                e = emotion::sad;
            }
            else if (s == "surprise") {
                e = emotion::surprise;
            }
            else {
                e = emotion::unknown;
            }
            return target.substr(0, emoPos);
        }
        map<string, float> termRelFreq;
        set<string>word;
    public:
        sentence() = default;
        explicit sentence(const string& s) :str(getEmotion(s)) {
            load(s);
        }
        ~sentence() = default;
        emotion getEmotion()const {
            if (e == emotion::unknown)
                throw invalid_argument("Emotion do not exist!");//
            return e;
        }
        string getStr()const {
            return str;
        }
        float getWordRelFreq(string s)const {
            return termRelFreq.at(s);
        }
        int getWordCnt()const {
            return wordCnt;
        }
        const set<string>& getWordSet() const {
            return word;
        }
        const map < string, float >& getTF()const {
            return termRelFreq;
        }
        friend bool operator<(const sentence& s1, const sentence& s2) { return s1.str < s2.str; }
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
            multiset<sentence>sentenceSet;
            map<string, int> wordFreqMap;
            string s;
            getline(infile, s);
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
            //�˺�data����
        }
        void TF_IDF_init() {
            if (sentenceCnt == 0 || wordCnt == 0)
                throw invalid_argument("classification.h::line 116:Data hasn't been initialized!");
            TF_IDF = vector<vector<float>>(sentenceCnt, vector<float>(wordCnt, 0));
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
        vector<float> encode(const sentence& s) {
            vector<float> code(wordCnt, 0);//�������ڼ������
            auto& wordset = s.getWordSet();
            int i;//�ҵ����ӵĵ�����word�����е�λ��
            for (auto w : wordset) {
                auto pos = find(word.begin(), word.end(), w);
                if (pos != word.end()) {
                    i = pos - word.begin();
                    code[i] = s.getWordRelFreq(w) * InvDocFreq[i];
                }
            }

            return code;
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

        vector<pair<sentence*, float>> distLP(const sentence& s, codeType st, float p = 1) {
            vector<float>code = encode(s);
            if (p <= 0)throw invalid_argument("line 185: p must be positive!");
            vector<pair<sentence*, float>> res(sentenceCnt);
            for (int j = 0; j < sentenceCnt; j++)
            {
                const auto& v1 = TF_IDF[j];
                res[j].first = &data[j];
                res[j].second = 0;
                float lhs, rhs;
                for (int k = 0; k < wordCnt; k++)
                {
                    if (st == codeType::one_hot) {
                        const auto& sent = data[j];
                        lhs = v1[k] == 0 ? 0 : sent.getWordRelFreq(word[k]) * sent.getWordCnt();
                        rhs = code[k] == 0 ? 0 : s.getWordCnt() * s.getWordRelFreq(word[k]);
                    }
                    else
                        lhs = v1[k], rhs = code[k];
                    if (lhs != 0 || rhs != 0) {
                        res[j].second += pow(abs(lhs - rhs), p);
                    }
                }
                res[j].second = pow(res[j].second, 1 / p);
            }
            return res;
        }
        vector<pair<sentence*, float>> distCosine(const sentence& s, codeType st) {
            vector<float>code = encode(s);
            vector<pair<sentence*, float>> res(sentenceCnt);
            for (int j = 0; j < sentenceCnt; j++)
            {
                const auto& v1 = TF_IDF[j];
                res[j].first = &data[j];
                res[j].second = 0;
                float normlhs = 0, normrhs = 0, lhs, rhs;
                for (int k = 0; k < wordCnt; k++)
                {
                    if (st == codeType::one_hot) {
                        const auto& sent = data[j];
                        lhs = v1[k] == 0 ? 0 : sent.getWordRelFreq(word[k]) * sent.getWordCnt();
                        rhs = code[k] == 0 ? 0 : s.getWordCnt() * s.getWordRelFreq(word[k]);
                    }
                    else
                        lhs = v1[k], rhs = code[k];
                    if (lhs != 0 || rhs != 0) {
                        res[j].second += lhs * rhs;
                        normlhs += lhs * lhs;
                        normrhs += rhs * rhs;
                    }
                }
                res[j].second = 1 - res[j].second / sqrt(normlhs * normrhs);
            }
            return res;
        }
        int sentenceCnt = 0;
        int wordCnt = 0;
        ~dataSet() = default;
    };

    class KNN :public dataSet
    {
    public:
        KNN() = default;
        KNN(ifstream& infile) :dataSet(infile) {}
        ~KNN() = default;
        KNN& setK(int _k) { k = _k; return *this; }
        KNN& setP(int _p) { p = _p; return *this; }
        KNN& setCodeType(codeType _ct) { ct = _ct; return *this; }
        emotion classify(const sentence& target) {
            vector<pair<sentence*, float>> vec;
            if (p == 0) {
                vec = distCosine(target, ct);
            }
            else
                vec = distLP(target, ct, p);
            sort(vec.begin(), vec.end(), [](const pair<sentence*, float>& p1, const pair<sentence*, float>& p2) {return p1.second < p2.second; });
            map<emotion, int>kElem;
            for (int i = 0; i < k; i++)
            {
                kElem[vec[i].first->getEmotion()]++;
            }
            auto res = std::max_element
            (
                kElem.begin(), kElem.end(),
                [](const pair<emotion, int>& p1, const pair<emotion, int>& p2) {
                    return p1.second < p2.second;
                }
            );
            return res->first;
        }

    private:
        int k = 1;
        int p = 1;
        codeType ct = codeType::one_hot;
    };
    KNN train(int max_k = 1, int max_p = 0);
    void test(KNN& k);

}
