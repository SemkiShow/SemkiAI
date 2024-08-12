#include "DatasetParser.h"

vector<string> files;

int ListFiles(char* path)
{
    DIR* directory = opendir(path);
    if (directory == NULL) {return 1;}
    struct dirent* file;
    file = readdir(directory);
    // cout << "Reading files in:";
    // cout << path << endl;
    while (file != NULL)
    {
        if (strcmp(file->d_name, ".") != 0 && strcmp(file->d_name, "..") != 0 && file->d_type == DT_REG)
        {
            char newPath[1024] = {0};
            strcat(newPath, path);
            strcat(newPath, "/");
            strcat(newPath, file->d_name);
            // cout << newPath << endl;
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

vector<string> Split(string input, char delimiter)
{
    string tmp;
    vector<string> output;
    for (int i = 0; i < input.length(); i++)
    {
        if (input[i] == delimiter)
        {
            output.push_back(tmp);
            tmp = "";
        }
        else
        {
            tmp += input[i];
        }
    }
    return output;
}

int ParseHTML(char* path, char* outputDirectory)
{
    ListFiles(path);
    cout << "There are a total of ";
    cout << files.size();
    cout << " files in the given path" << endl;
    cout << "Show all of them?(y/N) ";
    string answer;
    getline(cin, answer);
    for (auto& tmp : answer) {
        tmp = tolower(tmp);
    }
    if (/* answer.empty() ||  */answer == "y")
    {
        for (int i = 0; i < files.size(); i++)
        {
            cout << files[i] << endl;
        }
    }

    fstream input;
    fstream output;
    string file;
    string fileContent;
    char writePath[1024] = {0};
    vector<string> tmp;
    for (int i = 0; i < files.size(); i++)
    {
        input.open(files[i]);
        while (getline(input, file))
        {
            // cout << file << endl;
            fileContent += file;
            fileContent += "\n";
        }
        // cout << "------------------" << endl;
        // cout << fileContent << endl;
        input.close();

        tmp = Split(files[i], '/');
        strcat(writePath, outputDirectory);
        strcat(writePath, "/");
        // strcat(writePath, &tmp[tmp.size()]);
        output.open(writePath);
        cout << tmp[tmp.size()] << endl;

        file = "";
        fileContent = "";
        memset(writePath, 0, sizeof(writePath));
        output.close();
        tmp = vector<string>();
    }
    
    return 0;
}
