#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace  Eigen;
using namespace std;

// funzione che risolve il sistema tramite la decomposizione A = PLU con pivoting parziale che mi restituisca il vettore delle soluzioni e l'errore relativo
double PLUparziale (const MatrixXd& A, const VectorXd& b, const VectorXd& soluzione, Vector2d& xLU)
{
    xLU = A.partialPivLu().solve(b);
    double err_rel_LU = (soluzione-xLU).norm()/soluzione.norm();
    return err_rel_LU;
}

// funzione che risolve il sistema tramite la decomposizione P = QR che mi restituisca il vettore delle soluzioni e l'errore relativo
double QR (const MatrixXd& A, const VectorXd& b, const VectorXd& soluzione, Vector2d& xQR)
{
    xQR = A.householderQr().solve(b);
    double err_rel_QR = (soluzione-xQR).norm()/soluzione.norm();
    return err_rel_QR;
}

int main()
{
    // Inizializzo le matrici A e i vettori dei termini noti b di ogni sistema e il vettore delle soluzioni
    Matrix2d A1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02, 8.320502943378437e-01, -9.992887623566787e-01;
    Vector2d b1;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
    Matrix2d A2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    Vector2d b2;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
    Matrix2d A3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
    Vector2d b3;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
    Vector2d soluzione;
    soluzione << -1.0e+00, -1.0e+00;

    // Per ogni sistema uso le funzioni create in precedenza e mi calcolo gli errori relativi
    Vector2d x1LU;
    double err_rel_LU1 = PLUparziale(A1, b1, soluzione, x1LU);
    cout << scientific << setprecision(2) << "Soluzione sistema 1 usando PALU: [" << x1LU(0) << "; " << x1LU(1) <<
            "], con errore relativo: " << err_rel_LU1 << endl;
    Vector2d x1QR;
    double err_rel_QR1 = QR(A1, b1, soluzione, x1QR);
    cout << scientific << setprecision(2) << "Soluzione sistema 1 usando QR: [" << x1QR(0) << "; " << x1QR(1) <<
        "], con errore relativo: " << err_rel_QR1 << endl;

    Vector2d x2LU;
    double err_rel_LU2 = PLUparziale(A2, b2, soluzione, x2LU);
    cout << scientific << setprecision(2) << "Soluzione sistema 2 usando PALU: [" << x2LU(0) << "; " << x2LU(1) <<
        "], con errore relativo: " << err_rel_LU2 << endl;
    Vector2d x2QR;
    double err_rel_QR2 = QR(A2, b2, soluzione, x2QR);
    cout << scientific << setprecision(2) << "Soluzione sistema 2 usando QR: [" << x2QR(0) << "; " << x2QR(1) <<
        "], con errore relativo: " << err_rel_QR2 << endl;

    Vector2d x3LU;
    double err_rel_LU3 = PLUparziale(A3, b3, soluzione, x3LU);
    cout << scientific << setprecision(2) << "Soluzione sistema 3 usando PALU: [" << x3LU(0) << "; " << x3LU(1) <<
        "], con errore relativo: " << err_rel_LU3 << endl;
    Vector2d x3QR;
    double err_rel_QR3 = QR(A3, b3, soluzione, x3QR);
    cout << scientific << setprecision(2) << "Soluzione sistema 3 usando QR: [" << x3QR(0) << "; " << x3QR(1) <<
        "], con errore relativo: " << err_rel_QR3 << endl;

    return 0;
}
