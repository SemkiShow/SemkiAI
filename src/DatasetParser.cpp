#include "DatasetParser.hpp"

std::vector<std::string> files;

int ListFiles(char* path)
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

void ParseHTML(char* path, char* outputDirectory)
{
    ListFiles(path);
    std::cout << "There are a total of ";
    std::cout << files.size();
    std::cout << " files in the given path" << std::endl;
    std::cout << "Show all of them?(y/N) ";
    std::string answer;
    std::getline(std::cin, answer);
    for (auto& tmp : answer) {
        tmp = tolower(tmp);
    }
    if (/* answer.empty() ||  */answer == "y")
    {
        for (int i = 0; i < files.size(); i++)
        {
            std::cout << files[i] << std::endl;
        }
    }

    std::fstream input;
    std::fstream output;
    std::string file;
    std::string fileContent;
    char writePath[1024] = {0};
    std::vector<std::string> tmp;
    for (int i = 0; i < files.size(); i++)
    {
        input.open(files[i]);
        while (std::getline(input, file))
        {
            // std::cout << file << std::endl;
            fileContent += file;
            fileContent += "\n";
        }
        // std::cout << "------------------" << std::endl;
        // std::cout << fileContent << std::endl;
        input.close();

        tmp = Split(files[i], '/');
        strcat(writePath, outputDirectory);
        strcat(writePath, "/");
        // strcat(writePath, &tmp[tmp.size()]);
        output.open(writePath);
        std::cout << tmp[tmp.size()] << std::endl;

        file = "";
        fileContent = "";
        memset(writePath, 0, sizeof(writePath));
        output.close();
        tmp = std::vector<std::string>();
    }
}

std::vector<std::vector<std::string>> ParseCSV(std::string path)
{
    std::fstream datasetFile;
    datasetFile.open(path, std::ios::in);
    std::string buf = "";
    std::vector<std::string> datasetContents;
    while (std::getline(datasetFile, buf))
        datasetContents.push_back(buf);

    std::vector<std::vector<std::string>> output;
    for (int i = 0; i < datasetContents.size(); i++)
        output.push_back(Split(datasetContents[i], ','));

    return output;
}
