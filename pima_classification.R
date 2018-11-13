library(VIM)
library(e1071)
library(caret)
library(rpart)
library(C50)
library(class)
library(MASS)
library(randomForest)



#pima데이터의 경우, 0을 결측치 처리하므로
#na.string=0설정
pima<-read.csv(file.choose(), na.strings = '0')
str(pima)

#target은 0인 경우에 결측치가 아니므로 다시 복구
pima[is.na(pima$Target),]$Target<-0

#결측치 확인
missing<-aggr(pima)
summary(missing)

#class_mean imputation

#NPRG

classMean_NPRG <- tapply(pima[!is.na(pima$NPRG),]$NPRG, pima[!is.na(pima$NPRG),]$Target, mean)
classMean_NPRG
pima[is.na(pima$NPRG) & pima$Target==0,]$NPRG<-classMean_NPRG['0']
pima[is.na(pima$NPRG) & pima$Target==1,]$NPRG<-classMean_NPRG['1']

#PGC
classMean_PGC <- tapply(pima[!is.na(pima$PGC),]$PGC, pima[!is.na(pima$PGC),]$Target, mean)
classMean_PGC
pima[is.na(pima$PGC) & pima$Target==0,]$PGC<-classMean_PGC['0']
pima[is.na(pima$PGC) & pima$Target==1,]$PGC<-classMean_PGC['1']

#DBP
classMean_DBP <- tapply(pima[!is.na(pima$DBP),]$DBP, pima[!is.na(pima$DBP),]$Target, mean)
classMean_DBP
pima[is.na(pima$DBP) & pima$Target==0,]$DBP<-classMean_DBP['0']
pima[is.na(pima$DBP) & pima$Target==1,]$DBP<-classMean_DBP['1']

#BMI
classMean_BMI <- tapply(pima[!is.na(pima$BMI),]$BMI, pima[!is.na(pima$BMI),]$Target, mean)
classMean_BMI
pima[is.na(pima$BMI) & pima$Target==0,]$BMI<-classMean_BMI['0']
pima[is.na(pima$BMI) & pima$Target==1,]$BMI<-classMean_BMI['1']


# TSFT, SI2H의 속성은 결측치의 비율이 무척 높으므로
# Missing_value_indicator를 속성을 추가로 생성

#TSFT
#Missing_value_indicator생성
pima$TSFT_MI[is.na(pima$TSFT)]<-1
pima[is.na(pima$TSFT_MI),]$TSFT_MI<-0

#class_mean_imputation
classMean_TSFT <- tapply(pima[!is.na(pima$TSFT),]$TSFT, pima[!is.na(pima$TSFT),]$Target, mean)
classMean_TSFT
pima[is.na(pima$TSFT) & pima$Target==0,]$TSFT<-classMean_TSFT['0']
pima[is.na(pima$TSFT) & pima$Target==1,]$TSFT<-classMean_TSFT['1']

#SI2H
#Missing_value_indicator생성
pima$SI2H_MI[is.na(pima$SI2H)]<-1
pima[is.na(pima$SI2H_MI),]$SI2H_MI<-0

#class_mean_imputation
classMean_SI2H <- tapply(pima[!is.na(pima$SI2H),]$SI2H, pima[!is.na(pima$SI2H),]$Target, mean)
classMean_SI2H
pima[is.na(pima$SI2H) & pima$Target==0,]$SI2H<-classMean_SI2H['0']
pima[is.na(pima$SI2H) & pima$Target==1,]$SI2H<-classMean_SI2H['1']


pima$Target<-as.factor(pima$Target)



##여기까지 결측치 처리 완료
##이제 학습자료를 분할한다.
##target의 비율을 유지하는 층화샘플링을 이용하여 8:2분할.

pima0<-pima[pima$Target==0,]
pima1<-pima[pima$Target==1,]

set.seed(123)
Sample.N0_00<-sample(1:nrow(pima0),nrow(pima0)*0.8)
Sample.N0_01<-sample(1:nrow(pima1),nrow(pima1)*0.8)

pima_train<-rbind(pima0[Sample.N0_00,], pima1[Sample.N0_01,])
pima_test<-rbind(pima0[-Sample.N0_00,], pima1[-Sample.N0_01,])







##지금부터 여러 분류알고리즘을 이용하여
##각 알고리즘의 성능을 비교해본다.

#######################
#naive_bayesian
#######################

#fitting
fit.naive<-naiveBayes(Target~. , 
                      data=pima_train)

#pred_train
pred_naive_train<-predict(fit.naive,
                          newdata = pima_train,type='class')
confusionMatrix(pima_train$Target,
                pred_naive_train)


#pred_test
pred_naive_test<-predict(fit.naive,
                         newdata = pima_test,type='class')
confusionMatrix(pima_test$Target,
                pred_naive_test)


#######################
#decision tree_cart
#######################

#초모수 튜닝
set.seed(124)
cart_Hyper_parameter<-tune.rpart(Target~.,
                                 data = pima_train,
                                 minsplit = c(seq(10,20,by=0.5)),
                                 minbucket = c(seq(3,10,by=0.5)))
#튜닝된 초모수를 확인.
cart_Hyper_parameter$best.parameters

#fitting
fit_cart<-rpart(Target~.,
                data = pima_train,
                control = list(minsplit=cart_Hyper_parameter$best.parameters[1], minbucket=cart_Hyper_parameter$best.parameters[2]))


#pred_train
pred_cart_train<-predict(fit_cart,
                         newdata = pima_train, type = 'class')
confusionMatrix(pima_train$Target,
                pred_cart_train)
#pred_test
pred_cart_test<-predict(fit_cart,
                        newdata = pima_test,type='class')
confusionMatrix(pima_test$Target,
                pred_cart_test)








#######################
#decision tree_C5.0
#######################


#fitting
fit_c50<-C5.0(Target~.,
              data = pima_train,
              control = C5.0Control(minCases = 10))


#pred_train
pred_c50_train<-predict(fit_c50,
                       newdata = pima_train, type = 'class')
confusionMatrix(pima_train$Target,
                pred_c50_train)
#pred_test
pred_c50_test<-predict(fit_c50,
                       newdata = pima_test,type='class')
confusionMatrix(pima_test$Target,
                pred_c50_test)




#######################
#KNN
#######################


#초모수 튜닝
set.seed(125)
KNN_Hyper_parameter<-tune.knn(x=pima_train[,-9],
                              y=pima_train[,9], 
                              k=seq(5,19,by=1))


#튜닝된 초모수를 확인.
KNN_Hyper_parameter$best.parameters

#pred_train
pred_knn_train<-knn(pima_train,
                    pima_train,
                    cl = pima_train$Target,
                    k = KNN_Hyper_parameter$best.parameters[1])
confusionMatrix(pima_train$Target,
                pred_knn_train)

#pred_test
pred_knn_test<-knn(pima_train,
                   pima_test,
                   cl = pima_train$Target,
                   k = KNN_Hyper_parameter$best.parameters[1])
confusionMatrix(pima_test$Target,
                pred_knn_test)









#######################
#logit_reg
#######################

#fitting
fit_logit<-glm(Target~.,
               data = pima_train,
               family = binomial('logit'))
summary(fit_logit)

#pred_train
pred_logit_train<-predict(fit_logit,
                          newdata = pima_train, type = 'response')
pred_logit_train<-ifelse(pred_logit_train>0.5,1,0)
confusionMatrix(pima_train$Target,
                as.factor(pred_logit_train))
#pred_test
pred_logit_test<-predict(fit_logit,
                         newdata = pima_test,type= 'response')
pred_logit_test<-ifelse(pred_logit_test>0.5,1,0)
confusionMatrix(pima_test$Target,
                as.factor(pred_logit_test))






#######################
#logit_reg(stepwise)
#######################

#fitting
fit_null<-glm(Target~1,
              data = pima_train,
              family = binomial('logit'))
fit_logit_stepwise<-step(fit_null, direction = 'both', scope=list(upper=fit_logit))
summary(fit_logit_stepwise)

#pred_train
pred_logit_stepwise_train<-predict(fit_logit_stepwise,
                           newdata = pima_train, type = 'response')
pred_logit_stepwise_train<-ifelse(pred_logit_stepwise_train>0.5,1,0)
confusionMatrix(pima_train$Target,
                as.factor(pred_logit_stepwise_train))
#pred_test
pred_logit_stepwise_test<-predict(fit_logit_stepwise,
                          newdata = pima_test,type= 'response')
pred_logit_stepwise_test<-ifelse(pred_logit_stepwise_test>0.5,1,0)
confusionMatrix(pima_test$Target,
                as.factor(pred_logit_stepwise_test))









#######################
#Linear Discriminant Analysis(LDA)
#######################

#fitting
fit_lda<-lda(Target~. ,
             data=pima_train)

#pred_train
pred_lda_train<-predict(fit_lda,
                        newdata = pima_train)
confusionMatrix(pima_train$Target,
                pred_lda_train$class)
#pred_test
pred_lda_test<-predict(fit_lda,
                       newdata = pima_test)
confusionMatrix(pima_test$Target,
                pred_lda_test$class)







#######################
#svm_linear
#######################

#초모수 튜닝
set.seed(130)
svm_linear_Hyper_parameter<-tune.svm(Target~. ,
                                     data=pima_train,
                                     cost=c(seq(0.1,1,by=0.1),1:5),
                                     kernel="linear")
                              
#튜닝된 초모수를 확인.
svm_linear_Hyper_parameter$best.parameters

#fitting
fit_svm_linear<-svm(Target~.,
                    data = pima_train,
                    kernel="linear",
                    cost=svm_linear_Hyper_parameter$best.parameters[,1])
                    

#pred_train
pred_svm_linear_train<-predict(fit_svm_linear,
                               newdata = pima_train, type = 'class')
confusionMatrix(pima_train$Target,
                pred_svm_linear_train)
#pred_test
pred_svm_linear_test<-predict(fit_svm_linear,
                              newdata = pima_test,type='class')
confusionMatrix(pima_test$Target,
                pred_svm_linear_test)





#######################
#svm_polynomial
#######################

#초모수 튜닝
set.seed(131)
svm_poly_Hyper_parameter<-tune.svm(Target~. ,
                                   data=pima_train,
                                   cost=c(seq(0.1,1,by=0.1),1:5),
                                   degree=2:5,
                                   kernel="polynomial")

#튜닝된 초모수를 확인.
svm_poly_Hyper_parameter$best.parameters

#fitting
fit_svm_poly<-svm(Target~.,
                  data = pima_train,
                  kernel="polynomial",
                  cost=svm_poly_Hyper_parameter$best.parameters[,2],
                  degree=svm_poly_Hyper_parameter$best.parameters[,1])


#pred_train
pred_svm_poly_train<-predict(fit_svm_poly,
                             newdata = pima_train, type = 'class')
confusionMatrix(pima_train$Target,
                pred_svm_poly_train)
#pred_test
pred_svm_poly_test<-predict(fit_svm_poly,
                            newdata = pima_test,type='class')
confusionMatrix(pima_test$Target,
                pred_svm_poly_test)






#######################
#svm_raidal
#######################

#초모수 튜닝
set.seed(134)
svm_radial_Hyper_parameter<-tune.svm(Target~. ,
                                     data=pima_train,
                                     cost=c(seq(0.1,1,by=0.1),1:5),
                                     gamma=10^(-4:2),
                                     kernel="radial")

#튜닝된 초모수를 확인.
svm_radial_Hyper_parameter$best.parameters

#fitting
fit_svm_radial<-svm(Target~.,
                  data = pima_train,
                  kernel="radial",
                  cost=svm_radial_Hyper_parameter$best.parameters[,2],
                  gamma=svm_radial_Hyper_parameter$best.parameters[,1])


#pred_train
pred_svm_radial_train<-predict(fit_svm_radial,
                             newdata = pima_train, type = 'class')
confusionMatrix(pima_train$Target,
                pred_svm_radial_train)
#pred_test
pred_svm_radial_test<-predict(fit_svm_radial,
                            newdata = pima_test,type='class')
confusionMatrix(pima_test$Target,
                pred_svm_radial_test)







#######################
#random forest
#######################

#초모수 튜닝
set.seed(144)
random_forest_Hyper_parameter<-tune.randomForest(Target~.,
                                                 data=pima_train,
                                                 ntree=seq(50,150,by=10),
                                                 mtry = 3:5)

#튜닝된 초모수를 확인.
random_forest_Hyper_parameter$best.parameters

#fitting
fit_random_forest<-randomForest(Target~.,
                                data = pima_train,
                                ntree=random_forest_Hyper_parameter$best.parameters[,2],
                                mtry=random_forest_Hyper_parameter$best.parameters[,1],
                                do.trace=30,
                                nodesize=10,
                                importance=T)
                                

#pred_train
pred_random_forest_train<-predict(fit_random_forest,
                                  newdata = pima_train, type = 'class')
confusionMatrix(pima_train$Target,
                pred_random_forest_train)
#pred_test
pred_random_forest_test<-predict(fit_random_forest,
                                 newdata = pima_test,type='class')
confusionMatrix(pima_test$Target,
                pred_random_forest_test)



