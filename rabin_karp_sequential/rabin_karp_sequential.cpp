#include <fstream>
#include <iostream>
#include <sstream> 
#include<math.h>
#include <cctype>

using namespace std;

int alphabetLen = 26;
int mod = 23;

string readTextFromFile(string pathToFile);
int calculateHash(string text);
int moveHash(char oldChar, char newChar, int oldValue, int textLen);
bool compareText(int length, string text, string pattern);
int modulo(int x, int N);

int main()
{
    string text = readTextFromFile("test.txt");
    string pattern;

    cout << "Podaj wzorzec: ";
    cin >> pattern;

    cout << endl << "Tekst:" << endl;
    cout << text << endl;
    cout << "__________" << endl;

    cout << "Wzorzec:" << endl;
    cout << pattern << endl;
    cout << "__________" << endl;

    int hashOfPattern = calculateHash(pattern);
    int patternLength = pattern.length();
    int textLength = text.length();
    string pieceOfText = text.substr(0, patternLength);
    int hashOfPieceOfText = calculateHash(pieceOfText);

    if (hashOfPattern == hashOfPieceOfText) {
        if (compareText(patternLength, pieceOfText, pattern)) {
            cout << "Znaleziono od indeksu: 0" << endl;
        }
    }

    for (int i = 1; i <= textLength - patternLength; i++) {
        hashOfPieceOfText = moveHash(text[i - 1], text[i + patternLength - 1], hashOfPieceOfText, patternLength);
        pieceOfText = text.substr(i, patternLength);

        if (hashOfPattern == hashOfPieceOfText) {            
            if (compareText(patternLength, pieceOfText, pattern)) {
                cout << "Znaleziono od indeksu: " + to_string(i) << endl;
            }
        }       
    }

    return 0;
}

bool compareText(int length, string text, string pattern) {
    for (int i = 0; i < length; i++) {
        if (text[i] != pattern[i]) {
            return false;
        }
    }
    
    return true;
}

int calculateHash(string text) {
    int result = 0;
    int exponent = text.length() - 1;

    for (int i = 0; i < text.length(); i++) {
        result += (tolower(text[i]) - 'a') * pow(alphabetLen, exponent);
        exponent--;
    }

    return modulo(result, mod);
}

int modulo(int x, int N) {
    return (x % N + N) % N;
}

int moveHash(char oldChar, char newChar, int oldValue, int textLen) {
    int multiplier = (int) pow(alphabetLen, textLen - 1);
    
    int valueWithoutOldChar = (oldValue - (multiplier * (tolower(oldChar) - 'a')));
   
    int valueWithNewChar = valueWithoutOldChar * alphabetLen;    
    valueWithNewChar += (tolower(newChar) - 'a');

    return modulo(valueWithNewChar,mod);
}

string readTextFromFile(string pathToFile) {
    ifstream inFile;
    stringstream strStream;
        
    inFile.open(pathToFile);

    strStream << inFile.rdbuf();

    return strStream.str();
}
