

# cd ~/hw5

# make clean; make seq #2> make-stderr.out

# testCase=20
# ./nbody ./testcases/b$testCase.in out
# python3 ./validate.py out ./testcases/b$testCase.out





cd ~/hw5

make clean; make 2> make-stderr.out
RunFile=./hw5

testCase=20

inFile=./testcases/b$testCase.in

outFile=out.out
golden_outFile=./testcases/b$testCase.out

if [ -f "$RunFile" ]; then

    echo "==================================="
    echo "=            Run hw5             ="
    echo "==================================="

    export CUDA_VISIBLE_DEVICES=0,1
    
    ./$RunFile $inFile $outFile > run-stderr.out
    # nvprof --metrics ipc ./$RunFile $inFile $outFile > run-stderr.out

    echo "==================================="
    echo "=      Print run-stderr.out       ="
    echo "==================================="

    cat run-stderr.out


    echo "==================================="
    echo "=            Validate             ="
    echo "==================================="

    python3 ./validate.py $outFile $golden_outFile
    # rm $outFile

else

    echo "==================================="
    echo "=      Print make-stderr.out      ="
    echo "==================================="

    cat make-stderr.out    

fi


echo ""
echo ""