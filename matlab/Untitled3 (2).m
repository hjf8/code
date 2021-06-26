A=[1 2 3;2 4 5;3 5 6];
aij=[];
for j=1:3
    %for i=1:j-1
     %   aij=aij+A((i-1)*3-i*(i-1)/2+j);
    %end
    for i=j:3
        aij=A((j-1)*3-j*(j-1)/2+i);
        disp(aij)
    end
end

