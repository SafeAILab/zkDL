#include "bls12-381.cuh"
#include <iostream>

using namespace std;

int main()
{
  blstrs__scalar__Scalar x {4294967294, 1, 215042, 1485092858, 3971764213, 2576109551, 2898593135, 405057881}; 
  blstrs__scalar__Scalar y {{1, 2, 3, 4, 5, 6, 7, 8}};
  blstrs__scalar__Scalar z = x;
  z.val[0] -= 1;

  for (uint i = 0; i < 8; ++ i) cout << x.val[i] << endl;
  for (uint i = 0; i < 8; ++ i) cout << y.val[i] << endl;
  for (uint i = 0; i < 8; ++ i) cout << z.val[i] << endl;

  return 0;
}