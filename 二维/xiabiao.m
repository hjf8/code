function X=xiabiao()
X = sparse(10,10);
for i = 1:10
    for j = 1:10
        if i <= 5
            if (i+j == 5)
                X(i,j) = mod(i+j,5)+5;
            else
                X(i,j) = mod(i+j,5);
            end
        else
            X(i,j) = mod(i+j,5)+5;
        end
    end
end
X = full(X);
        
            
            
            
            