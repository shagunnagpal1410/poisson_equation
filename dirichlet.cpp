#include<iostream>
#include<vector>
#include<map>
#include<cmath>
#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<Eigen/SparseLU>
#include<fstream>
#include<string>
using namespace std;
using namespace Eigen;
typedef Triplet<double> Tri;
struct point{
    double x,y;
    int voxel;
    point(double i, double j) {
        x=i, y=j;
        voxel=0;
    }
    bool operator<(const point& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return voxel < other.voxel;
    }
};
int find_voxel(double x, double y, double x_min, double x_max, int voxels_inrow, double radius) {
    return int((x-x_min)/radius) + int((y)/radius)*voxels_inrow;
}
int find_rownumber(int voxel_number, int voxels_inrow) {
    return voxel_number%voxels_inrow;
}
int find_columnnumber(int voxel_number, int voxels_inrow) {
    return int(voxel_number/voxels_inrow);
}
double gaussian_weight_function(point Ni, point p0, double radius) {
    double distance=(pow(Ni.x-p0.x,2)+pow(Ni.y-p0.y,2))/(radius*radius);
    if (distance<=1) {
        return exp(-6.25*distance);
    }
    else {
        return 0.0;
    }
}
const double PI = acos(-1.0);

double analytical_solution(double x, double y, int N_terms = 100) {
    double sum = 0.0;
    for (int n = 1; n <= N_terms; n++) {
        int k = 2 * n - 1;
        double lambda = k * PI;

        double term = sinh(lambda * (1 - y)) + sinh(lambda * y);
        term /= sinh(lambda);
        term *= sin(lambda * x);
        term /= (k * k * k);  // (2n - 1)^3

        sum += term;
    }

    double result = (1 - x) * x - (8.0 / (PI * PI * PI)) * sum;
    return result;
}

int main() {
    //initializing some parameters------------------Starting-------------------------------------------------------
    double L=1;
    double B=1;
    double dx=0.01;
    double dy=0.01;
    vector<point> previous_domain;
    double x_min=0;
    double x_max=1;
    double y_min=0;
    double y_max=1;
    double radius=0.12;
    int voxels_inrow=int((x_max-x_min)/radius)+1;
    int voxels_incolumn=int(1/radius)+1;
    //initializing some parameters------------------Ending---------------------------------------------------------
    //adding points in the domain-------------------Starting-------------------------------------------------------
    int n=ceil((x_max-x_min)/dx);
    int m=ceil((y_max-y_min)/dy);
    for (int i=0; i<=n; i++) {
        for (int j=0; j<=m; j++) {
            point p1=point(i*dx,j*dy);
            p1.voxel=find_voxel(p1.x,p1.y,x_min,x_max,voxels_inrow,radius);
            previous_domain.push_back(p1);
        }
    }
    int max_voxels=voxels_incolumn*voxels_inrow;
    vector<vector<point>> points_insidevoxel(max_voxels);
    for (int i=0; i<previous_domain.size(); i++) {
        point p0=previous_domain[i];
        points_insidevoxel[p0.voxel].push_back(p0);
    }
    //adding points in the domain-------------------Ending---------------------------------------------------------
    //finding neighbour voxels of each voxel--------Starting-------------------------------------------------------
    vector<vector<int>> neighbour_voxels(max_voxels);
    for(int i=0; i<max_voxels; i++) {
        int find_x=find_rownumber(i, voxels_inrow);
        int find_y=find_columnnumber(i, voxels_inrow);
        for (int diffx=-1; diffx<=1; diffx++) {
            for (int diffy=-1; diffy<=1; diffy++) {
                if (find_x+diffx>=0 && find_x+diffx<=voxels_inrow-1 && diffy+find_y>=0 && find_y+diffy<=voxels_incolumn-1) {
                            neighbour_voxels[i].push_back((find_y+diffy)*voxels_inrow+(find_x+diffx));
                }
            }
        }
    }
    //finding neighbour voxels of each voxel--------Ending--------------------------------------------------------
    map<point,double> known_Temperature;
        int row_index=0;
        map<point,int> rownumber_Temperature;
        for (int p=0; p<previous_domain.size(); p++) {
            point p0=previous_domain[p];
            if (p0.x==x_min || p0.x==x_max || p0.y==y_min || p0.y==y_max) {
                known_Temperature[p0]=0;
            }
            else {
                rownumber_Temperature[p0]=row_index;
                row_index+=1;
            }
        }
        int unknown_Temperature=rownumber_Temperature.size();
        SparseMatrix<double> A(unknown_Temperature,unknown_Temperature);
        VectorXd rhs(unknown_Temperature);
        vector<Tri> coefficients;
        for (auto it: rownumber_Temperature) {
            point p0=it.first;
            int rownumber=it.second;
            vector<point> known_neighbours;
            vector<point> unknown_neighbours; 
            int voxel_num=p0.voxel;
            for(int i=0; i<neighbour_voxels[voxel_num].size(); i++) {
                int neighbour_voxel=neighbour_voxels[voxel_num][i];
                for (int j=0; j<points_insidevoxel[neighbour_voxel].size(); j++) {
                    point Ni= points_insidevoxel[neighbour_voxel][j];
                    double distance=pow(Ni.x-p0.x,2)+pow(Ni.y-p0.y,2);
                    if (distance>0 && distance<=pow(radius,2)) {
                        if (known_Temperature.find(Ni)==known_Temperature.end()) {
                            unknown_neighbours.push_back(Ni);
                        }
                        else {
                            known_neighbours.push_back(Ni);
                        }
                    }
                }
            }
            int total_neighbours=known_neighbours.size()+unknown_neighbours.size();
            MatrixXd M(total_neighbours,6);
            MatrixXd W=MatrixXd :: Zero(total_neighbours, total_neighbours);
            int number=0;
            for(int i=0; i<unknown_neighbours.size(); i++) {
                point Ni=unknown_neighbours[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(number,0)=1, M(number,1)=Dx, M(number,2)=Dy, M(number,3)=(Dx*Dx)/2, M(number,4)=(Dy*Dy)/2,
                M(number,5)=Dx*Dy;
                W(number,number)=gaussian_weight_function(Ni,p0,radius);
                number+=1;
            }
            for(int i=0; i<known_neighbours.size(); i++) {
                point Ni=known_neighbours[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(number,0)=1, M(number,1)=Dx, M(number,2)=Dy, M(number,3)=(Dx*Dx)/2, M(number,4)=(Dy*Dy)/2,
                M(number,5)=Dx*Dy;
                W(number,number)=gaussian_weight_function(Ni,p0,radius);
                number+=1;
            }
            RowVectorXd L(6);
            VectorXd CT;
            double constant;
            L<<0,0,0,1,1,0;
            constant=-2.0;
            MatrixXd MTWM = M.transpose() * W * M;
            CT=(L * MTWM.ldlt().solve(M.transpose() * W));
            for (int i=0; i<unknown_neighbours.size(); i++) {
                point Ni=unknown_neighbours[i];
                coefficients.push_back(Tri(rownumber,rownumber_Temperature[Ni] , CT(i)));
            }
            for (int i=0; i<known_neighbours.size(); i++){
                point Ni=known_neighbours[i];
                constant-=CT(unknown_neighbours.size()+i)*known_Temperature[Ni];
            }
            rhs(rownumber)=constant;
        }
        A.setFromTriplets(coefficients.begin(),coefficients.end());
        SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
        solver.compute(A);
        VectorXd answer=solver.solve(rhs);
        map<point,double> Temperature;
        for (int p=0; p<previous_domain.size(); p++) {
            point p0=previous_domain[p];
            if (known_Temperature.find(p0)==known_Temperature.end()) {
                Temperature[p0]=answer(rownumber_Temperature[p0]);
            }
            else {
                Temperature[p0]=known_Temperature[p0];
            }
        }
        cout<<"Temperature calculated!!!"<<endl;
        ofstream fout("Temperature.csv");
        fout<<"X"<<","<<"Y"<<","<<"Temperature"<<"\n";
        for (auto it: Temperature) {
            point p1=it.first;
            double temp=it.second;
            fout<<p1.x<<","<<p1.y<<","<<temp<<"\n";
        }
        fout.close();
        int N = 100; // number of terms in series

    ofstream fout("PoissonSeriesSolution.csv");
    fout << "X,Y,Temperature\n";

    for (double x = 0.0; x <= 1.0 + 1e-6; x += dx) {
        for (double y = 0.0; y <= 1.0 + 1e-6; y += dx) {
            double u = analytical_solution(x, y, N);
            fout << x << "," << y << "," << u << "\n";
        }
    }

    fout.close();
    cout << "Saved to PoissonSeriesSolution.csv\n";
    return 0;
}