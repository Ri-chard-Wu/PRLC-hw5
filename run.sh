

cd ~/hw5

make clean; make seq #2> make-stderr.out
# RunFile=./nbody

testCase=20
./nbody ./testcases/b$testCase.in out
python3 ./validate.py out ./testcases/b$testCase.out

# testCase=03

# inFile=./testcases/case$testCase.in

# outFile=out.out
# golden_outFile=./testcases/case$testCase.out

# if [ -f "$RunFile" ]; then

#     echo "==================================="
#     echo "=            Run hw4             ="
#     echo "==================================="

#     export CUDA_VISIBLE_DEVICES=0
    
#     ./$RunFile $inFile $outFile 2> run-stderr.out
#     # nvprof --metrics ipc ./$RunFile $inFile $outFile > run-stderr.out

#     echo "==================================="
#     echo "=      Print run-stderr.out       ="
#     echo "==================================="

#     cat run-stderr.out


#     echo "==================================="
#     echo "=            Validate             ="
#     echo "==================================="

#     # ./validation $outFile $golden_outFile
#     # rm $outFile

# else

#     echo "==================================="
#     echo "=      Print make-stderr.out      ="
#     echo "==================================="

#     cat make-stderr.out    

# fi


# echo ""
# echo ""