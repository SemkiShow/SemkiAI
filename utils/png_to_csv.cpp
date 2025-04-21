#include "pnglib.hpp"
#include <iostream>
#include <fstream>
#include <string.h>
#include <dirent.h>
#include <vector>
#include <cstdint>
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

int ListFiles(const char* path)
{
    DIR* directory = opendir(path);
    if (directory == NULL) {return 1;}
    struct dirent* file;
    file = readdir(directory);
    // std::cout << "Reading files in:";
    // std::cout << path << std::endl;
    while (file != NULL)
    {
        if (strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0 && file->d_type == DT_REG)
        {
            char newPath[1024] = {0};
            strcat(newPath, path);
            strcat(newPath, "/");
            strcat(newPath, file->d_name);
            // std::cout << newPath << std::endl;
            files.push_back(newPath);
        }
        if (strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0 && file->d_type == DT_DIR)
        {
            char newPath[256] = {0};
            strcat(newPath, path);
            strcat(newPath, "/");
            strcat(newPath, file->d_name);
            ListFiles(newPath);
        }
        file = readdir(directory);
    }

    closedir(directory);
    return 0;
}

int main()
{
    std::cout << "Input the source path:\n";
    std::string source = "";
    std::getline(std::cin, source);

    ListFiles(source.c_str());

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
