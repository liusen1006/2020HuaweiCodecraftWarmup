#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#define NUM_THREADS 8

using namespace std;

//矩阵操作
struct Matrix {
    typedef vector<float> Mat1;
    typedef vector<vector<float>> Mat2;
};
void out(Matrix::Mat1 mat) {
    for (auto &x : mat) {
        cout << x << " ";
    }
    cout << "\n";
}
void out(Matrix::Mat2 mat) {
    for (auto &x : mat) {
        out(x);
    }
}
//点乘
static float Dot(const Matrix::Mat1 &mat1, const Matrix::Mat1 &mat2) {
    int n = mat1.size();
    float ans = 0;
    for (int i = 0; i < n; i += 16) {
        for (int j = 0; j < 16; j++) {
            ans += mat1[i + j] * mat2[i + j];
        }
    }
    return ans;
};
//乘法
static Matrix::Mat1 operator*(const Matrix::Mat2 &mat1,
                              const Matrix::Mat1 &mat2) {
    int n = mat1.size();
    Matrix::Mat1 mat;
    for (const auto &x : mat1) {
        mat.emplace_back(Dot(x, mat2));
    }
    return mat;
};
static Matrix::Mat2 operator*(const Matrix::Mat1 &mat1,
                              const Matrix::Mat1 &mat2) {
    int n = mat1.size(), m = mat2.size(), id = 0;
    Matrix::Mat2 mat(n);
    for (const auto &x : mat1) {
        for (const auto &y : mat2) {
            mat[id].emplace_back(x * y);
        }
        id++;
    }
    return mat;
};
//转置
static Matrix::Mat2 T(const Matrix::Mat2 &mat1) {
    int n = mat1.size(), m = mat1[0].size();
    Matrix::Mat2 mat(m, Matrix::Mat1(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            mat[j][i] = mat1[i][j];
        }
    }
    return mat;
};
static Matrix::Mat1 operator-(const Matrix::Mat1 &mat1,
                              const Matrix::Mat1 &mat2) {
    int n = mat1.size();
    Matrix::Mat1 mat(n);
    for (int i = 0; i < n; i++) {
        mat[i] = mat1[i] - mat2[i];
    }
    return mat;
}
static Matrix::Mat2 operator*(const Matrix::Mat2 &mat1,
                              const Matrix::Mat2 &mat2) {
    Matrix::Mat2 mat2T = T(mat2);
    int n = mat1.size(), m = mat2[0].size();
    Matrix::Mat2 mat(n, Matrix::Mat1(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            mat[i][j] = Dot(mat1[i], mat2T[j]);
        }
    }
    return mat;
};

char answer[40000];
string trainFile;
string testFile;
string answerFile;
string predictOutFile;
int testLineSize = 6000; //测试集一行字符数
int feature = 400;    //特征数
int trainNum = 1580;  //参与训练样本数
int predictNum = 100000;
int featureId;
float pLabel0;
float pLabel1;
int items;
vector<vector<float>> mu[2];
vector<vector<int>> trainLabel;
vector<vector<float>> delta;
vector<vector<pair<int, int>>> threadStart;

//预存x * (0.1)^y
float pw[10][5] = {0, 0,   0,    0,     0,      1, 0.1, 0.01, 0.001, 0.0001,
                   2, 0.2, 0.02, 0.002, 0.0002, 3, 0.3, 0.03, 0.003, 0.0003,
                   4, 0.4, 0.04, 0.004, 0.0004, 5, 0.5, 0.05, 0.005, 0.0005,
                   6, 0.6, 0.06, 0.006, 0.0006, 7, 0.7, 0.07, 0.007, 0.0007,
                   8, 0.8, 0.08, 0.008, 0.0008, 9, 0.9, 0.09, 0.009, 0.0009};
float pwChar[255][5]; 

void getReadId(char *buffer) {
    int now = 0, pre, threadId = 0, j = 0,
        p = (trainNum + NUM_THREADS - 1) / NUM_THREADS, circle = 0;

    for (int i = 0; i < NUM_THREADS; i++) {
        if (now + p <= trainNum) {
            threadStart.emplace_back(vector<pair<int, int>>(p));
            now += p;
        } else {
            threadStart.emplace_back(vector<pair<int, int>>(trainNum - now));
        }
    }
    now = 0;
    for (int i = 0; i < trainNum; ++i) {
        pre = now;
        now += testLineSize;
        while (buffer[now] != '\n') ++now;
        threadStart[threadId][j++] = make_pair(pre, now - 1);
        circle++;
        if (circle == p) {
            threadId++;
            j = 0;
            circle = 0;
        }
        now++;
    }
}

void LoadChar(char *buffer, int &pid, int &start, int &end) {
    int now = start, id = 0, r = 0;
    float num = 0, sum = 0;
    bool flag = false;

    int type = buffer[end] - '0';
    trainLabel[pid][type]++;
    while (id < feature) {
        if (buffer[now] == '-') {
            now++;
            num = pwChar[buffer[now]][0] + pwChar[buffer[now + 2]][1] +
                  pwChar[buffer[now + 3]][2] + pwChar[buffer[now + 4]][3];
            mu[type][pid][id++] -= num;
            sum -= num;
        } else {
            num = pwChar[buffer[now]][0] + pwChar[buffer[now + 2]][1] +
                  pwChar[buffer[now + 3]][2] + pwChar[buffer[now + 4]][3];
            mu[type][pid][id++] += num;
            sum += num;
        }
        now += 6;
    }
    delta[pid][type] += sum;
}

void threadLoadData(char *buffer, int pid) {
    for (auto &x : threadStart[pid]) {
        LoadChar(buffer, pid, x.first, x.second);
    }
}

void loadTrainData() {
    struct stat sb;
    int fd = open(trainFile.c_str(), O_RDONLY);
    fstat(fd, &sb);
    char *buffer =
        (char *)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    getReadId(buffer);
    vector<thread> td(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        td[i] = thread(&threadLoadData, buffer, i);
    }
    for (auto &t : td) {
        t.join();
    }
    close(fd);
}

void threadPredict(char *buffer, int pid, int start, int end, int lineSize) {
    int id, initId = start * lineSize, nowId, up, now = 0, j, k, r;
    float sum;
    int fd = open(predictOutFile.c_str(), O_RDWR | O_CREAT, 0666);
    char *answer =
        (char *)mmap(NULL, 40000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    for (int i = start; i < end; i += items) {
        up = items;
        if (i + items > end) up = end - i;
        for (k = 0; k < up; k++) {
            id = 0;
            sum = pLabel1;
            for (j = 0; j < featureId; j += 60) {
                nowId = initId + j;
                for (r = 0; r < 60; r += 6) {
                    sum += (pwChar[buffer[nowId + r]][0] +
                            pwChar[buffer[nowId + r + 2]][1] +
                            pwChar[buffer[nowId + r + 3]][2]) *
                           mu[0][0][id];
                    ++id;
                }
            }
            answer[(i + k) << 1 | 1] = '\n';
            answer[(i + k) << 1] = sum > 0 ? '1' : '0';
            initId += lineSize;
        }
    }
    munmap(answer, 40000);
}

void loadTestData(const string &file, int &lineSize, int pid) {
    struct stat sb;
    int fd = open(file.c_str(), O_RDONLY);
    fstat(fd, &sb);
    char *buffer =
        (char *)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    int linenum = sb.st_size / lineSize;
    int pre = 0, line = (linenum + NUM_THREADS - 1) / NUM_THREADS;
    pre = line * pid;

    threadPredict(buffer, pid, pre, min(pre + line, linenum), lineSize);
}


void train() {
    for (int i = 1; i < NUM_THREADS; i++) {
        trainLabel[0][0] += trainLabel[i][0];
        delta[0][0] += delta[i][0];
        delta[0][1] += delta[i][1];
    }
    delta[0][0] += 1;
    delta[0][1] += 1;

    pLabel0 = 1.0 * trainLabel[0][0] / trainNum;
    pLabel1 = 1.0 - pLabel0;
    pLabel0 = log(pLabel0);
    pLabel1 = log(pLabel1);

    float al0 = log(1.0 / delta[0][0]), al1 = log(1.0 / delta[0][1]);
    int nowId;
    for (int i = 0; i < feature; i += items) {
        for (int k = 0; k < items; k++) {
            nowId = i + k;
            for (int j = 1; j < NUM_THREADS; j++) {
                mu[0][0][nowId] += mu[0][j][nowId];
                mu[1][0][nowId] += mu[1][j][nowId];
            }

            mu[0][0][nowId] = log(mu[0][0][nowId] + 1) + al0;
            mu[1][0][nowId] = log(mu[1][0][nowId] + 1) + al1;
        }
    }
    mu[0][0] = mu[1][0] - mu[0][0];
    pLabel1 -= pLabel0;
}

void judge() {
    vector<int> answer, result;
    int x, cor = 0;
    ifstream fin(answerFile);
    while (fin) {
        fin >> x;
        answer.emplace_back(x);
    }
    fin.close();
    ifstream fin2(predictOutFile);
    while (fin2) {
        fin2 >> x;
        result.emplace_back(x);
    }
    fin2.close();
    for (int i = 0; i < answer.size(); i++) {
        if (answer[i] == result[i]) cor++;
    }
    cout << "准确率: " << 1.0 * cor / answer.size() << "\n";
}
void init() {
    items = 64 / sizeof(float);
    for (int i = 0; i < NUM_THREADS; i++) {
        mu[0].emplace_back(vector<float>(feature, 0));
        mu[1].emplace_back(vector<float>(feature, 0));
        delta.emplace_back(vector<float>(2, 0));
        trainLabel.emplace_back(vector<int>(2, 0));
    }
    FILE *fd = fopen(predictOutFile.c_str(), "w");
    for (int i = 0; i < 40000; i += 4) {
        char ch[4] = {' ', ' ', ' ', ' '};
        fwrite(ch, 4, 1, fd);
    }
    fclose(fd);
    for (int i = '0'; i <= '9'; i++) {
        for (int j = 0; j < 5; j++) {
            pwChar[i][j] = pw[i - '0'][j];
        }
    }
    featureId = feature * 6;

    loadTrainData();
    train();
}

void Bayes(string trainF, string testF, string predictOutF, string answerF) {
    trainFile = trainF;
    testFile = testF;
    predictOutFile = predictOutF;
    answerFile = answerF;
    init();
}
int main(int argc, char *argv[]) {
    string trainFile = "../data/train_data.txt";
    string testFile = "../data/test_data.txt";
    string predictFile = "../data/result.txt";
    string answerFile = "../data/answer.txt";

    // string trainFile = "/data/train_data.txt";
    // string testFile = "/data/test_data.txt";
    // string predictFile = "/projects/student/result.txt";
    // string answerFile = "/projects/student/answer.txt";

    Bayes(trainFile, testFile, predictFile, answerFile);

    pid_t fk[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    fk[1] = fork();
    if (fk[1]) fk[2] = fork();
    if (fk[2]) fk[3] = fork();
    if (fk[3]) fk[4] = fork();
    if (fk[4]) fk[5] = fork();
    if (fk[5]) fk[6] = fork();
    if (fk[6]) fk[7] = fork();

    int pid = 0;
    for (int i = 1; i <= 7; i++) {
        if (!fk[i]) {
            pid = i;
            break;
        }
    }

    if (pid <= 7) {
        loadTestData(testFile, testLineSize, pid);
        exit(0);
    }
    // judge();
    return 0;
}
