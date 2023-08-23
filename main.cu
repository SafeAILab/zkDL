#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include <iostream>
#include <iomanip>
#include "timer.hpp"

using namespace std;

ostream& operator<<(ostream& os, const Fr_t& x)
{
  os << "0x" << std::hex;
  for (uint i = 8; i > 0; -- i)
  {
    os << std::setfill('0') << std::setw(8) << x.val[i - 1];
  }
  return os << std::dec << std::setw(0) << std::setfill(' ');
}


int main(int argc, char *argv[])
{
  uint size = stoi(argv[1]);

  Fr_t* cpu_data = new Fr_t[size];
  for (uint i = 0; i < size; ++ i)
  {
    cpu_data[i].val[0] = i;
  }

  cout << "size=" << size << endl;
  Timer timer;
  timer.start();
  FrTensor t1(size);
  timer.stop();
  // cout << timer.getTotalTime() << endl;
  timer.reset();

  cout << "======== Testing constructors ========" << endl;
  timer.start();
  FrTensor t2(size);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  timer.start();
  FrTensor t3(size, cpu_data);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing G1Affine constructor ========" << endl;
  timer.start();
  G1TensorAffine gt1(size);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  timer.start();
  G1TensorAffine gt2(size, G1Affine_generator);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  timer.start();
  G1TensorAffine gt3 = gt2;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing G1Jacobian constructor ========" << endl;
  timer.start();
  G1TensorJacobian gt4(size);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  timer.start();
  G1TensorJacobian gt5(size, G1Jacobian_generator);
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  timer.start();
  G1TensorJacobian gt6 = gt5;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  timer.start();
  G1TensorJacobian gt7 = gt3;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing G1Jacobian operators ========" << endl;
  timer.start();
  auto gt8 = gt5 + gt2;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  timer.start();
  auto gt9 = gt8 - gt3;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  timer.start();
  gt9 += gt7;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  timer.start();
  gt8 += G1Affine_generator;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing sum ========" << endl;
  timer.start();
  auto s = t3.sum();
  timer.stop();
  cout << timer.getTotalTime() << endl;
  cout << s << endl;
  timer.reset();
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing copy ========" << endl;
  timer.start();
  FrTensor t4 = t3;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  cout << t4(1) << endl;
  timer.reset();
  cout << "======== End of test ========" << endl << endl;

  srand(time(NULL));
  int randidx = rand() % size;
  cout << "Testing at index "<< randidx << endl << endl;

  timer.start();
  FrTensor t5 = t3 + t4;
  timer.stop();
  // cout << timer.getTotalTime() << endl;
  timer.reset();

  cout << "======== Testing operator+ ========" << endl;
  timer.start();
  FrTensor t6 = t4 + t5;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();

  cout << t4(randidx) << endl;
  cout << t5(randidx) << endl;
  cout << t6(randidx) << endl;
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing unary operator- ========" << endl;
  timer.start();
  FrTensor t7 = -t6;
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  cout << t7(randidx) << endl;
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing operator += ========" << endl;
  timer.start();
  t7 += {2, 0, 0, 0, 0, 0, 0, 0};
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  cout << t7(randidx) << endl;
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing operator -= ========" << endl;
  timer.start();
  t7 -= {1, 0, 0, 0, 0, 0, 0, 0};
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  cout << t7(randidx) << endl;
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing mont ========" << endl;
  timer.start();
  t7.mont();
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  cout << t7(randidx) << endl;
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing unmont ========" << endl;
  timer.start();
  t7.unmont();
  timer.stop();
  cout << timer.getTotalTime() << endl;
  timer.reset();
  cout << t7(randidx) << endl;
  cout << "======== End of test ========" << endl << endl;

  cout << "======== Testing mul ========" << endl;
  timer.start();
  auto t8 = t5 * t7;
  t8.mont();
  timer.stop();
  cout << timer.getTotalTime() << endl;
  cout << t5(randidx) << endl;
  cout << t7(randidx) << endl;
  cout << t8(randidx) << endl;
  cout << "======== End of test ========" << endl << endl;

  delete[] cpu_data;
  return 0;
}