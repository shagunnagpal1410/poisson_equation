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

double analytical_solution(double x, double y, int N) {
    const double PI = acos(-1.0);
    double sum=0.0;
    for (int n=0; n<=N; n++) {
        double num=(pow(-1,n+1)*cos((2*n+1)*PI*0.5*y)*cosh((2*n+1)*PI*0.5*x));
        double deno=pow((2*n+1),3)*cosh((2*n+1)*0.5*PI);
        double val=(num/deno);
        sum=sum+val;
    }
    return pow(1-y,2)+((32.0*sum)/(pow(PI,3)));
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
    map<point, int> identity;
    for (int p=0; p<previous_domain.size(); p++) {
        point p0=previous_domain[p];
        identity[p0]=p;
    }
    int number_ofpoints=identity.size();
    SparseMatrix<double> letsSolve(number_ofpoints,number_ofpoints);
    VectorXd rhs(number_ofpoints);
    vector<Tri> coefficient;
    for (int p=0; p<previous_domain.size(); p++) {
        cout<<"point- "<<p+1<<" done"<<endl;
        point p0=previous_domain[p];
        vector<point> neighbours_point;
        int voxel_num=p0.voxel;
        for (int i=0; i<neighbour_voxels[voxel_num].size(); i++) {
            int neighbour_voxel=neighbour_voxels[voxel_num][i];
            for (int j=0; j<points_insidevoxel[neighbour_voxel].size(); j++) {
                point Ni=points_insidevoxel[neighbour_voxel][j];
                double distance=pow(Ni.x-p0.x,2)+pow(Ni.y-p0.y,2);
                if (distance>0 && distance<pow(radius,2)) {
                    neighbours_point.push_back(Ni);
                }
            }
        }
        int total_neighbours=neighbours_point.size();
        if (p0.x==x_min && p0.y==y_max) {
            MatrixXd M(total_neighbours+3,6);
            MatrixXd W =MatrixXd :: Zero(total_neighbours+3,total_neighbours+3);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(i,0)=1, M(i,1)=Dx, M(i,2)=Dy, M(i,3)=(Dx*Dx)/2, M(i,4)=(Dy*Dy)/2, M(i,5)=(Dx*Dy);
                W(i,i)=gaussian_weight_function(Ni,p0,radius);
            }
            M(total_neighbours,0)=0, M(total_neighbours,1)=0, M(total_neighbours,2)=0, M(total_neighbours,3)=1, M(total_neighbours,4)=1
            ,M(total_neighbours,5)=0;
            W(total_neighbours,total_neighbours)=1;
            M(total_neighbours+1,0)=1, M(total_neighbours+1,1)=0, M(total_neighbours+1,2)=0, M(total_neighbours+1,3)=0, M(total_neighbours+1,4)=0
            ,M(total_neighbours+1,5)=0;
            W(total_neighbours+1,total_neighbours+1)=1;
            M(total_neighbours+2,0)=0, M(total_neighbours+2,1)=1, M(total_neighbours+2,2)=0, M(total_neighbours+2,3)=0, M(total_neighbours+2,4)=0
            ,M(total_neighbours+2,5)=0;
            W(total_neighbours+2,total_neighbours+2)=1;
            MatrixXd MTWM=M.transpose()*W*M;
            MatrixXd MTW=M.transpose()*W;  
            MatrixXd A=MTWM.ldlt().solve(MTW); 
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                coefficient.push_back(Tri(identity[p0],identity[Ni], A(0,i)));
            }
            coefficient.push_back(Tri(identity[p0],identity[p0],-1));
            rhs(identity[p0])=(A(0,total_neighbours)*2.0);
            cout << "Point: (" << p0.x << "," << p0.y << "), A(0,total_neighbours): " << A(0, total_neighbours) << endl;
        }
        else if (p0.x==x_max && p0.y==y_min) {
            MatrixXd M(total_neighbours+3,6);
            MatrixXd W =MatrixXd :: Zero(total_neighbours+3,total_neighbours+3);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(i,0)=1, M(i,1)=Dx, M(i,2)=Dy, M(i,3)=(Dx*Dx)/2, M(i,4)=(Dy*Dy)/2, M(i,5)=(Dx*Dy);
                W(i,i)=gaussian_weight_function(Ni,p0,radius);
            }
            M(total_neighbours,0)=0, M(total_neighbours,1)=0, M(total_neighbours,2)=0, M(total_neighbours,3)=1, M(total_neighbours,4)=1
            ,M(total_neighbours,5)=0;
            W(total_neighbours,total_neighbours)=1;
            M(total_neighbours+1,0)=1, M(total_neighbours+1,1)=0, M(total_neighbours+1,2)=0, M(total_neighbours+1,3)=0, M(total_neighbours+1,4)=0
            ,M(total_neighbours+1,5)=0;
            W(total_neighbours+1,total_neighbours+1)=1;
            M(total_neighbours+2,0)=0, M(total_neighbours+2,1)=0, M(total_neighbours+2,2)=1, M(total_neighbours+2,3)=0, M(total_neighbours+2,4)=0
            ,M(total_neighbours+2,5)=0;
            W(total_neighbours+2,total_neighbours+2)=1;
            MatrixXd MTWM=M.transpose()*W*M;
            MatrixXd MTW=M.transpose()*W;  
            MatrixXd A=MTWM.ldlt().solve(MTW); 
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                coefficient.push_back(Tri(identity[p0],identity[Ni], A(0,i)));
            }
            coefficient.push_back(Tri(identity[p0],identity[p0],-1));
            rhs(identity[p0])=(A(0,total_neighbours)*2.0);
            cout << "Point: (" << p0.x << "," << p0.y << "), A(0,total_neighbours): " << A(0, total_neighbours) << endl;
        }
        else if (p0.x==x_min && p0.y==y_min)  {
            MatrixXd M(total_neighbours+3,6);
            MatrixXd W =MatrixXd :: Zero(total_neighbours+3,total_neighbours+3);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(i,0)=1, M(i,1)=Dx, M(i,2)=Dy, M(i,3)=(Dx*Dx)/2, M(i,4)=(Dy*Dy)/2, M(i,5)=(Dx*Dy);
                W(i,i)=gaussian_weight_function(Ni,p0,radius);
            }
            M(total_neighbours,0)=0, M(total_neighbours,1)=0, M(total_neighbours,2)=0, M(total_neighbours,3)=1, M(total_neighbours,4)=1
            ,M(total_neighbours,5)=0;
            W(total_neighbours,total_neighbours)=1;
            M(total_neighbours+1,0)=0, M(total_neighbours+1,1)=1, M(total_neighbours+1,2)=0, M(total_neighbours+1,3)=0, M(total_neighbours+1,4)=0
            ,M(total_neighbours+1,5)=0;
            W(total_neighbours+1,total_neighbours+1)=1;
            M(total_neighbours+2,0)=0, M(total_neighbours+2,1)=0, M(total_neighbours+2,2)=1, M(total_neighbours+2,3)=0, M(total_neighbours+2,4)=0
            ,M(total_neighbours+2,5)=0;
            W(total_neighbours+2,total_neighbours+2)=1;
            MatrixXd MTWM=M.transpose()*W*M;
            MatrixXd MTW=M.transpose()*W;  
            MatrixXd A=MTWM.ldlt().solve(MTW); 
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                coefficient.push_back(Tri(identity[p0],identity[Ni], A(0,i)));
            }
            coefficient.push_back(Tri(identity[p0],identity[p0],-1));
            rhs(identity[p0])=(A(0,total_neighbours)*2.0);
            cout << "Point: (" << p0.x << "," << p0.y << "), A(0,total_neighbours): " << A(0, total_neighbours) << endl;
        }
        else if (p0.x==x_max || p0.y==y_max) {
            MatrixXd M(total_neighbours+2,6);
            MatrixXd W=MatrixXd :: Zero(total_neighbours+2,total_neighbours+2);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(i,0)=1, M(i,1)=Dx, M(i,2)=Dy, M(i,3)=(Dx*Dx)/2, M(i,4)=(Dy*Dy)/2, M(i,5)=(Dx*Dy);
                W(i,i)=gaussian_weight_function(Ni,p0,radius);
            }
            M(total_neighbours,0)=0, M(total_neighbours,1)=0, M(total_neighbours,2)=0, M(total_neighbours,3)=1, M(total_neighbours,4)=1
            ,M(total_neighbours,5)=0;
            W(total_neighbours,total_neighbours)=1;
            M(total_neighbours+1,0)=1, M(total_neighbours+1,1)=0, M(total_neighbours+1,2)=0, M(total_neighbours+1,3)=0, M(total_neighbours+1,4)=0
            ,M(total_neighbours+1,5)=0;
            W(total_neighbours+1,total_neighbours+1)=1;
            MatrixXd MTWM=M.transpose()*W*M;
            MatrixXd MTW=M.transpose()*W;  
            MatrixXd A=MTWM.ldlt().solve(MTW);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                coefficient.push_back(Tri(identity[p0],identity[Ni], A(0,i)));
            }
            coefficient.push_back(Tri(identity[p0],identity[p0],-1));
            rhs(identity[p0])=(A(0,total_neighbours)*2.0);
            cout << "Point: (" << p0.x << "," << p0.y << "), A(0,total_neighbours): " << A(0, total_neighbours) << endl;

        }
        else if (p0.x==x_min) {
            MatrixXd M(total_neighbours+2,6);
            MatrixXd W=MatrixXd :: Zero(total_neighbours+2,total_neighbours+2);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(i,0)=1, M(i,1)=Dx, M(i,2)=Dy, M(i,3)=(Dx*Dx)/2, M(i,4)=(Dy*Dy)/2, M(i,5)=(Dx*Dy);
                W(i,i)=gaussian_weight_function(Ni,p0,radius);
            }
            M(total_neighbours,0)=0, M(total_neighbours,1)=0, M(total_neighbours,2)=0, M(total_neighbours,3)=1, M(total_neighbours,4)=1
            ,M(total_neighbours,5)=0;
            W(total_neighbours,total_neighbours)=1;
            M(total_neighbours+1,0)=0, M(total_neighbours+1,1)=1, M(total_neighbours+1,2)=0, M(total_neighbours+1,3)=0, M(total_neighbours+1,4)=0
            ,M(total_neighbours+1,5)=0;
            W(total_neighbours+1,total_neighbours+1)=1;
            MatrixXd MTWM=M.transpose()*W*M;
            MatrixXd MTW=M.transpose()*W;  
            MatrixXd A=MTWM.ldlt().solve(MTW);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                coefficient.push_back(Tri(identity[p0],identity[Ni], A(0,i)));
            }
            coefficient.push_back(Tri(identity[p0],identity[p0],-1));
            rhs(identity[p0])=(A(0,total_neighbours)*2.0);
            cout << "Point: (" << p0.x << "," << p0.y << "), A(0,total_neighbours): " << A(0, total_neighbours) << endl;
        }
        else if (p0.y==y_min) {
            MatrixXd M(total_neighbours+2,6);
            MatrixXd W=MatrixXd :: Zero(total_neighbours+2,total_neighbours+2);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(i,0)=1, M(i,1)=Dx, M(i,2)=Dy, M(i,3)=(Dx*Dx)/2, M(i,4)=(Dy*Dy)/2, M(i,5)=(Dx*Dy);
                W(i,i)=gaussian_weight_function(Ni,p0,radius);
            }
            M(total_neighbours,0)=0, M(total_neighbours,1)=0, M(total_neighbours,2)=0, M(total_neighbours,3)=1, M(total_neighbours,4)=1
            ,M(total_neighbours,5)=0;
            W(total_neighbours,total_neighbours)=1;
            M(total_neighbours+1,0)=0, M(total_neighbours+1,1)=0, M(total_neighbours+1,2)=1, M(total_neighbours+1,3)=0, M(total_neighbours+1,4)=0
            ,M(total_neighbours+1,5)=0;
            W(total_neighbours+1,total_neighbours+1)=1;
            MatrixXd MTWM=M.transpose()*W*M;
            MatrixXd MTW=M.transpose()*W;  
            MatrixXd A=MTWM.ldlt().solve(MTW);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                coefficient.push_back(Tri(identity[p0],identity[Ni], A(0,i)));
            }
            coefficient.push_back(Tri(identity[p0],identity[p0],-1));
            rhs(identity[p0])=(A(0,total_neighbours)*2.0);
            cout << "Point: (" << p0.x << "," << p0.y << "), A(0,total_neighbours): " << A(0, total_neighbours) << endl;
        }
        else {
            MatrixXd M(total_neighbours+1,6);
            MatrixXd W=MatrixXd :: Zero(total_neighbours+1,total_neighbours+1);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                double Dx=Ni.x-p0.x, Dy=Ni.y-p0.y;
                M(i,0)=1, M(i,1)=Dx, M(i,2)=Dy, M(i,3)=(Dx*Dx)/2, M(i,4)=(Dy*Dy)/2, M(i,5)=(Dx*Dy);
                W(i,i)=gaussian_weight_function(Ni,p0,radius);
            }
            M(total_neighbours,0)=0, M(total_neighbours,1)=0, M(total_neighbours,2)=0, M(total_neighbours,3)=1, M(total_neighbours,4)=1
            ,M(total_neighbours,5)=0;
            W(total_neighbours,total_neighbours)=1;
            MatrixXd MTWM=M.transpose()*W*M;
            MatrixXd MTW=M.transpose()*W;  
            MatrixXd A=MTWM.ldlt().solve(MTW);
            for (int i=0; i<total_neighbours; i++) {
                point Ni=neighbours_point[i];
                coefficient.push_back(Tri(identity[p0],identity[Ni], A(0,i)));
            }
            coefficient.push_back(Tri(identity[p0],identity[p0],-1));
            rhs(identity[p0])=(A(0,total_neighbours)*cos(M_PI*p0.x));
            cout << "Point: (" << p0.x << "," << p0.y << "), A(0,total_neighbours): " << A(0, total_neighbours) << endl;

        }
}
   letsSolve.setFromTriplets(coefficient.begin(), coefficient.end());

// STEP 2: Solve the system using Eigen's SparseLU
SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
solver.analyzePattern(letsSolve);
solver.factorize(letsSolve);

// Check if factorization succeeded
if (solver.info() != Success) {
    cout << "[ERROR] Matrix factorization failed!" << endl;
    return -1;
}

// Solve the system
VectorXd answer = solver.solve(rhs);

// Check if the solve succeeded
if (solver.info() != Success) {
    cout << "[ERROR] Linear system solve failed!" << endl;
    return -1;
}

// Optional debug: log solution stats
cout << "RHS max = " << rhs.maxCoeff() << ", min = " << rhs.minCoeff() << endl;
cout << "Answer max = " << answer.maxCoeff() << ", min = " << answer.minCoeff() << endl;

    cout<<"Temperature calculated!!!"<<endl;
    ofstream fout("Temperature2.csv");
    fout<<"X"<<","<<"Y"<<","<<"Temperature"<<"\n";
    for (int p=0; p<number_ofpoints; p++) {
        point p1=previous_domain[p];
        double temp=answer(p);
        fout<<p1.x<<","<<p1.y<<","<<temp<<"\n";
    }
    fout.close();
    ofstream fout1("PoissonSeriesSolution2.csv");
    fout1<<"X"<<","<<"Y"<<","<<"Temperature"<<"\n";
    int N=100;
    for (double x = 0.0; x <= 1.0 + 1e-6; x += dx) {
        for (double y = 0.0; y <= 1.0 + 1e-6; y += dx) {
            double u = analytical_solution(x, y,N);
            fout1<<x<<","<<y<<","<<u<<"\n";
        }
    }
    fout1.close();
    cout << "Saved to PoissonSeriesSolution.csv\n";
    return 0;
}