#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>

#include <experimental/filesystem>

using std::experimental::filesystem::create_directory;
using std::experimental::filesystem::remove_all;

void makeCSV(float **u, size_t X, size_t Y, float t, size_t N);
float **makeArray2D(size_t rows, size_t cols);
void freeArray2D(float **array2D, size_t rows, size_t cols);

float fi(float x, float y)
{
    float e = 0.05;
    float _x = 0.5;
    float _y = 0.4;
    if ((_x - e <= x && _x + e >= x) && (_y - e <= y && _y + e >= y))
        return 0.1;
    return 0;
}

int main(int argc, char *argv[])
{
    remove_all("res");
    create_directory("res");

    const float xmax = 1.0;
    const float ymax = 1.0;

    float tau = 0.1;
    float tmax = 5.0;
    float h = 0.1;
    float c = 1.0;

    int opt;
    while ((opt = getopt(argc, argv, "t:d:h:c:")) != -1)
    {
        switch (opt)
        {
        case 't':
        {
            tmax = std::atof(optarg);
            break;
        }
        case 'd':
        {
            tau = std::atof(optarg);
            break;
        }
        case 'h':
        {
            h = std::atof(optarg);
            break;
        }
        case 'c':
        {
            c = std::atof(optarg);
            break;
        }
        default:
            break;
        }
    }

    int N = (int)(xmax / h);

    float r = (c * c * tau * tau) / (h * h);

    float **U = makeArray2D(N, N);
    float **oldU = makeArray2D(N, N);
    float **futureU = makeArray2D(N, N);

#pragma omp for schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            // Начальное и граничное условие = 0
            oldU[i][j] = fi((i + 1) * h, (j + 1) * h);

            U[i][j] = 0.0;
            futureU[i][j] = 0.0;
        }

    makeCSV(oldU, N, N, 0, 0);

    // 1 tau
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            U[i][j] = oldU[i][j] + 0.5 * r * (oldU[i - 1][j] + oldU[i + 1][j] + oldU[i][j - 1] + oldU[i][j + 1] - 4.0 * oldU[i][j]);

    makeCSV(U, N, N, tau, 1);

    size_t counter = 2;
    // next tau
    for (float t = 2 * tau; t <= tmax; t += tau)
    {
#pragma omp for schedule(static)
        for (int i = 1; i < N - 1; i++)
            for (int j = 1; j < N - 1; j++)
                futureU[i][j] = 2.0 * U[i][j] - oldU[i][j] + r * (U[i - 1][j] + U[i + 1][j] + U[i][j - 1] + U[i][j + 1] - 4.0 * U[i][j]);

#pragma omp for schedule(static)
        for (int i = 1; i < N - 1; i++)
            for (int j = 1; j < N - 1; j++)
            {
                oldU[i][j] = U[i][j];
                U[i][j] = futureU[i][j];
            }

        makeCSV(U, N, N, tau * counter, counter);
        counter++;
    }

    freeArray2D(oldU, N, N);
    freeArray2D(U, N, N);
    freeArray2D(futureU, N, N);

    return 0;
}

void makeCSV(float **u, size_t X, size_t Y, float t, size_t N)
{
    std::ofstream outStream("res/" + std::to_string(N) + ".txt", std::ios_base::out);
    outStream << t << "\n";
    for (size_t i = 0; i < Y; ++i)
    {
        for (size_t j = 0; j < X; ++j)
        {
            outStream << u[i][j];
            if (j != Y - 1)
                outStream << " ";
        }
        outStream << std::endl;
    }
    outStream.close();
}

float **makeArray2D(size_t rows, size_t cols)
{
    float *data = new float[rows * cols];
    float **array2D = new float *[rows];

    for (size_t i = 0; i < rows; ++i)
        array2D[i] = &(data[i * cols]);

    return array2D;
}

void freeArray2D(float **array2D, size_t rows, size_t cols)
{
    delete[] & (array2D[0][0]);
    delete[] array2D;
}