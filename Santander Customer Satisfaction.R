library(xgboost)
library(Matrix) 

set.seed(1234) #selecciono una semilla para generar número aleatorios. Puedes usar cualquier número en el argumento, 1234, 2016, tu fecha de nacimiento. Usamos esta función para 
#hacer correr simulaciones y asegurarnos que todos los resultados son reproducibles. En el ejemplo de aquí te explica qué hace http://stackoverflow.com/questions/13605271/reasons-for-using-the-set-seed-function


train <- read.csv("/home/domingo/Desktop/train.csv")     #cargo fichero train (modifica la ruta para adaptarla a tus directorios)
test <- read.csv("/home/domingo/Desktop/input/test.csv") #cargo fichero test  (modifica la ruta para adaptarla a tus directorios)

##### Removing IDs 

train$ID <- NULL #hago la columna ID del train nula.
test.id <- test$ID #declaro el dataset test.id como la columna Id de test
test$ID <- NULL #hago la columna ID del test nula.

##### Extracting TARGET 

train.y <- train$TARGET #declaro el dataset train.y como la columna Target de train
train$TARGET <- NULL  #hago la columna target del train  nula

##### 0 count per line 
count0 <- function(x) {     #count0 es lo que caga la función que defino como function con un solo argumento x que sumala cantidad de ceros por línea.
  return( sum(x == 0) ) 
}
train$n0 <- apply(train, 1, FUN=count0) #la función apply come un dataset, una dirección dentro del dataset y por último una función.
test$n0 <- apply(test, 1, FUN=count0)  #estas funciones generan en train y test una nueva columna con el número de 0 por fila.

##### Removing constant features
cat("\n## Removing the constants features.\n") 
for (f in names(train)) 
{ if (length(unique(train[[f]])) == 1) 
{ cat(f, "is constant in train. We delete it.\n") 
  train[[f]] <- NULL 
  test[[f]] <- NULL } } 
##### Removing identical features 
features_pair <- combn(names(train), 2, simplify = F) toRemove <- c() 
for(pair in features_pair) { f1 <- pair[1] f2 <- pair[2] 
if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) 
  { if (all(train[[f1]] == train[[f2]])) { cat(f1, "and", f2, "are equals.\n") toRemove <- c(toRemove, f2) } } } 

feature.names <- setdiff(names(train), toRemove) 

train <- train[, feature.names] test <- test[, feature.names] 
train$TARGET <- train.y 
train <- sparse.model.matrix(TARGET ~ ., data = train) 
dtrain <- xgb.DMatrix(data=train, label=train.y) watchlist <- list(train=dtrain) 
param <- list( objective = "binary:logistic", booster = "gbtree", eval_metric = "auc", eta = 0.02, max_depth = 5, subsample = 0.7, colsample_bytree = 0.7 ) 
clf <- xgb.train( params = param, data = dtrain, nrounds = 551, verbose = 2, watchlist = watchlist, maximize = FALSE ) 
test$TARGET <- -1 
test <- sparse.model.matrix(TARGET ~ ., data = test) 
preds <- predict(clf, test) submission <- data.frame(ID=test.id, TARGET=preds) 
cat("saving the submission file\n") write.csv(submission, "submission.csv", row.names = F)