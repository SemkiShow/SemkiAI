#include "pnglib.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include <vector>
#include <cstdint>
#include <string>
using namespace puzniakowski::png;

std::vector<std::string> files;

std::vector<std::string> Split(std::string input, char delimiter)
{
    std::vector<std::string> output;
    output.push_back("");
    int index = 0;
    for (int i = 0; i < input.size(); i++)
    {
        if (input[i] == delimiter)
        {
            index++;
            output.push_back("");
            continue;
        }
        output[index] += input[i];
    }
    return output;
}

int ListFiles(std::string path)
{
    struct stat sb;
 
    // Looping until all the items of the directory are
    // exhausted
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
 
        // Converting the path to const char * in the
        // subsequent lines
        std::filesystem::path outfilename = entry.path();
        std::string outfilename_str = outfilename.string();
        const char* outfilename_char = outfilename_str.c_str();
 
        // Testing whether the path points to a
        // non-directory or not If it does, displays path
        if (stat(outfilename_char, &sb) == 0 && !(sb.st_mode & S_IFDIR))
            files.push_back(outfilename_str);
        else
            ListFiles(outfilename_str);
    }
    return 0;
}
int main()
{
    std::cout << "Input the source path:\n";
    std::string source = "";
    std::getline(std::cin, source);

    ListFiles(source);

    std::fstream output;
    output.open("output.csv", std::ios::out);
    std::string file = "";
    std::string fileContent = "";
    int8_t r,g,b,a;
    unsigned int color;
    pngimage_t inputImage;
    std::string buf = "";

    for (int k = 0; k < files.size(); k++)
    {
        // if (stoi(Split(Split(files[k], '/').end()[-1], '.')[0]) >= 10000) continue;
        inputImage = read_png_file(files[k]);
        buf += Split(Split(files[k], '/').end()[-2], '.')[0] + ',';
        
        for (int i = 0; i < inputImage.height; i++)
        {
            for (int j = 0; j < inputImage.width; j++)
            {
                color = inputImage.get(j, i);
                getRGBAFromColor(color, &r, &g, &b, &a);
                buf += std::to_string((uint8_t)a) + ',';
            }
        }
        buf.pop_back();
        buf += '\n';
        if (k % 10 == 0)
            std::cout << '\r' << "Converting...    " << (k * 1000 / files.size() / 10.0) << "% (" << k << "/" << files.size() << ")";
    }
    std::cout << "\n";
    output << buf;
    output.close();
}
