#include"classification.h"
#include"reggression.h"
KNN_CLASSIFICATION::KNN KNN_CLASSIFICATION::train(int max_k, int max_p)
{
    cout << "********************************************" << endl
        << "KNN_CLASSIFICATION:training..." << endl;
    string s;
    ifstream train_set("lab1_data\\classification_dataset\\train_set.csv");
    KNN k(train_set);
    train_set.close();
    codeType bestCt;
    ofstream train_res("classifi_train_res.csv");
    int K, P;
    float CR = 0;
    codeType CT;
    train_res << "k,p,codeType,accuracy" << endl;
    for (int i = 0; i < 2; i++)
    {
        int bestK = 1, bestP = 1;
        codeType bestCt, currCT = (codeType)i;
        cout << "********************************************" << endl
            << "KNN_CLASSIFICATION:mode:" << codeTypeStr[(int)currCT] << endl;
        k.setCodeType(currCT);
        float bestCR = 0;
        for (int i = 1; i < max_k; i++)
        {
            k.setK(i);
            for (int j = 0; j < max_p; j++)
            {
                k.setP(j);
                ifstream validation_set("lab1_data\\classification_dataset\\validation_set.csv");
                int correct = 0, total = 0;
                getline(validation_set, s);
                while (getline(validation_set, s)) {
                    auto sent = sentence(s);
                    if (sent.getEmotion() == k.classify(sent)) {
                        correct++;
                    }
                    total++;
                }
                cout << "k: " << i << ", p: " << j << ", accuracy: " << (float)correct / total << endl;
                train_res << i << ',' << j << ',' << codeTypeStr[(int)currCT] << ',' << (float)correct / total << endl;
                validation_set.close();
                if (bestCR < (float)correct / total) {
                    bestK = i, bestP = j, bestCR = (float)correct / total, bestCt = currCT;
                    K = bestK, P = bestP, CT = bestCt, CR = bestCR;
                }
            }
        }
        cout << "best K: " << bestK << ", best P: " << bestP << ", best accuracy: " << bestCR << endl;
    }
    k.setK(K).setP(P).setCodeType(CT);
    cout << "********************************************" << endl
        << "KNN_CLASSIFICATION::train over."
        << "best K: " << K << ", best P: " << P << ", best accuracy: " << CR << ", best codeType: " << codeTypeStr[(int)CT] << endl;
    train_res.close();
    return k;
}
void KNN_CLASSIFICATION::test(KNN_CLASSIFICATION::KNN& k) {
    cout << "********************************************" << endl
        << "KNN_CLASSIFICATION:testing..." << endl;
    string s;
    ifstream test_set("lab1_data\\classification_simple_test.csv");
    ofstream outfile("17338233_zhenggehan_KNN_classification1.csv");
    getline(test_set, s);
    while (getline(test_set, s)) {
        s = s.substr(s.find(',') + 1);
        auto sent = sentence(s);
        outfile << sent.getStr() << ',' << emoStr[(int)k.classify(sent)] << endl;
    }
    test_set.close();
    outfile.close();
    cout << "KNN_CLASSIFICATION:test over." << endl;
}

KNN_REGRESSION::KNN KNN_REGRESSION::train(int max_k, int max_p)
{
    cout << "********************************************" << endl
        << "KNN_CLASSIFICATION:training..." << endl;
    string s;
    ifstream train_set("lab1_data\\regression_dataset\\train_set.csv");
    ofstream train_res("regress_train_res.csv");
    KNN_REGRESSION::KNN k(train_set);
    train_set.close();
    int K, P;
    float CR;
    codeType CT;
    train_res << "k,p,codeType,accuracy" << endl;
    for (int i = 0; i < 2; i++)
    {
        int bestK = 1, bestP = 0;
        codeType bestCt, currCT = (codeType)i;
        cout << "********************************************" << endl
            << "KNN_REGRESSION:mode:" << codeTypeStr[(int)currCT] << endl;
        float bestCR = 0;
        k.setCodeType(currCT);
        for (int i = 1; i < max_k; i++)
        {
            k.setK(i);
            for (int j = 0; j < max_p; j++) {
                k.setP(j);
                ifstream validation_set("lab1_data\\regression_dataset\\validation_set.csv");
                float totalCOR = 0, total = 0;
                getline(validation_set, s);
                while (getline(validation_set, s)) {
                    auto sent = KNN_REGRESSION::sentence(s);
                    totalCOR += sent.correlate(k.classify(sent));
                    total += 1;
                }
                if (total <= 0)throw overflow_error("divided by zero!");
                float corrRate = (float)totalCOR / total;
                train_res << i << ',' << j << ',' << codeTypeStr[(int)currCT] << ',' << (float)corrRate << endl;
                cout << "k: " << i << ", p: " << j << ", accuracy: " << (float)corrRate << endl;
                validation_set.close();
                if (bestCR < corrRate) {
                    bestK = i, bestP = j, bestCR = (float)corrRate, bestCt = currCT;
                    K = bestK, P = bestP, CT = bestCt, CR = bestCR;
                }
            }
        }
        cout << "best K: " << bestK << ",best P: " << bestP << ", best corrRate: " << bestCR << endl;
    }
    cout << "********************************************" << endl;
    cout << "KNN_CLASSIFICATION::train over. "
        << "best K: " << K << ",best P: " << P << ", best corrRate: " << CR << ", best codeType: " << codeTypeStr[(int)CT] << endl;
    k.setK(K).setP(P).setCodeType(CT);
    train_res.close();
    return k;
}

KNN_REGRESSION::KNN KNN_REGRESSION::train1(int max_k, int max_p)
{
    cout << "********************************************" << endl
        << "KNN_CLASSIFICATION:training..." << endl;
    string s;
    ifstream train_set("lab1_data\\regression_dataset\\train_set.csv");
    ofstream train_res("regress_train_res.csv");
    KNN_REGRESSION::KNN k(train_set);
    train_set.close();
    int K, P;
    float CR;
    codeType CT;
    for (int i = 0; i < 2; i++)
    {
        codeType bestCt, currCT = (codeType)i;
        cout << "********************************************" << endl
            << "KNN_REGRESSION:mode:" << codeTypeStr[(int)currCT] << endl;
        train_res << "k,p,codeType,accuracy" << endl;
        int bestK = 1, bestP = 0;
        float bestCR = 0;
        k.setCodeType(currCT);
        for (int i = 1; i < max_k; i++)
        {
            k.setK(i);
            for (int j = 0; j < max_p; j++) {
                vector<array<float, 6>>vec1, vec2;
                k.setP(j);
                ifstream validation_set("lab1_data\\regression_dataset\\validation_set.csv");
                getline(validation_set, s);
                while (getline(validation_set, s)) {
                    auto sent = KNN_REGRESSION::sentence(s);
                    vec1.push_back(sent.getEmotion());
                    vec2.push_back(k.classify(sent));
                }
                float corrRate = fullCOR(vec1, vec2);
                train_res << i << ',' << j << ',' << codeTypeStr[(int)currCT] << ',' << (float)corrRate << endl;
                cout << "k: " << i << ", p: " << j << ", accuracy: " << (float)corrRate << endl;
                validation_set.close();
                if (bestCR < corrRate) {
                    bestK = i, bestP = j, bestCR = (float)corrRate, bestCt = currCT;
                    K = bestK, P = bestP, CT = bestCt, CR = bestCR;
                }
            }
        }
        cout << "best K: " << bestK << ",best P: " << bestP << ", best corrRate: " << bestCR << endl;
    }
    cout << "********************************************" << endl
        << "KNN_REGRESSION::train over. "
        << "best K: " << K << ",best P: " << P << ", best corrRate: " << CR << ", best codeType: " << codeTypeStr[(int)CT] << endl;
    k.setK(K).setP(P).setCodeType(CT);
    train_res.close();
    return k;
}
void KNN_REGRESSION::test(KNN_REGRESSION::KNN& k) {
    cout << "********************************************" << endl
        << "KNN_REGRESSION:testing..." << endl;
    string s;
    ifstream test_set("lab1_data\\regression_simple_test.csv");
    ofstream outfile("17338233_zhenggehan_KNN_regression1.csv");
    getline(test_set, s);
    int index = 0;
    outfile << "textid,anger,disgust,fear,joy,sad,surprise" << endl;
    while (getline(test_set, s)) {
        index++;
        auto first_comma = s.find_first_of(',');
        auto last_comma = s.find_last_of(',');
        s = s.substr(first_comma + 1, last_comma - first_comma - 1);
        auto sent = KNN_REGRESSION::sentence(s);
        auto res = k.classify(sent);
        outfile << index
            << ',' << res[(int)emotion::anger]
            << ',' << res[(int)emotion::disgust]
            << ',' << res[(int)emotion::fear]
            << ',' << res[(int)emotion::joy]
            << ',' << res[(int)emotion::sad]
            << ',' << res[(int)emotion::surprise]
            << endl;
    }
    test_set.close();
    outfile.close();
    cout << "KNN_REGRESSION::test over." << endl;
}