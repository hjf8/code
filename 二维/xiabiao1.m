function x=xiabiao()
x=sparse(10,4);
for i=1:10
    for j=1:4
    if i<=5
        if (i+j==5)
            x(i,j)=mod(i+j,5)+5;
        else
            x(i,j)=mod(i+j,5);
        end
    else if i>=6&i<=10
            if (i+j==5)
                x(i,j)=mod(i+j,5)+5+5;
            else 
               x(i,j)=mod(i+j,5)+5;
            end
        end
    end
    end
end
