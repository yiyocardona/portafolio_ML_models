

## Funciones de exploración

#########################################

#Dropping columns, quitar columnas

#col1= columna que quiero eliminar tiene que ser ingresada dentro de comillas

def alg_drop(df,col1):
    df.drop([col1],axis=1)

##########################################

#Algoritmo para contar nulls

def alg_contar_null(df):
    null_columns=df.columns[df.isnull().any()]
    data_imdb[null_columns].isnull().sum()


##########################################

# ALgoritmo para eliminar NAs

def alg_drop_na(df):
    df.dropna()

##########################################

#Reemplaza strings dentro de una columna, todo va escrito con comillas

#col1= columna que vamos a modificar
#str1 = string que vamos a reemplazar
#str2= string que vamos a poner en vez de la otra

def alg_replace(df,col1,str1,str2)
    df[col1] = df[col1].str.replace(str1, str2)

##########################################

## Histograma

##Column = la columna que va a hacer display

def alg_histogram(df,column):
    sns.distplot(df[column])

##########################################

## Correlacion


def alg_corr(df):
    plt.subplots(figsize=(20,15))

    corr = df.corr()


    sns.heatmap(corr)

##########################################

## Barplot

#col1 = columna que quiero hacer display en x
#col2= columna en y

def alg_barplot(df,col1,col2):
    ax = sns.barplot(x=col1, y=col2, data=df)

##########################################

##scatterplot

# col1= columna x
# col2 = columna y

def alg_scatter(df,col1,col2):
    sns.scatterplot(x="col1", y="col2", data=df)

###########################################



##########################################

##export csv

#name = el nombre de salida del csv

def alg_export(df,name):
    df.to_csv('name.csv')

##########################################

#Convertidor de 4 columnas en Dummies

#col1= en comillas
#col2= en comillas
#col3= en comillas
#col4= en comillas


def alg_dummies(col1, col2, col3, col4,df):
    detdum = []
    cat_vars = [ucol1, ucol2,
           ucol3, ucol4]


    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(df[var], prefix=var)
        detdum=pd.concat([df, cat_list], axis=1, sort=False)
        df = detdum

    data_vars=df.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    detdum=df[to_keep]



##########################################

### PCA

##Comp = numero de componentes para el PCA
## df = dataframe
## Label = numero de labels para el algoritmo

def alg_pca(comp,df,label):
    pca = PCA(n_components=comp)


    pc = pca.fit_transform(df)


    pc_df = pd.DataFrame(data = pc,
            columns = ['PC1', 'PC2','PC3'])
    pc_df['Cluster'] = label
    pc_df.head()

###########################################3

### LDA

## Feat= son los features que uno va a predecir
## Label = numero de labels para el algoritmo


def alg_lda(feat,label):

    sc = StandardScaler()
    features_std = sc.fit_transform(feat)


    lda = LDA()
    ld = lda.fit_transform(features_std, label)
    lda_df = pd.DataFrame(data = ld,
            columns = ['LDA1', 'LDA2'])
    lda_df['Cluster'] = label

    print('Exactitud del LDA en el set de entrenamiento: {:.2f}'
         .format(lda.score(features_std, label)))

    lda_df.head()

###################################################

def alg_svc(svc, param, X, y):
    clf = svc
    clf.fit(X, y)

    plt.clf()
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # Circle out the test data
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X.iloc[:, 0].min()
    x_max = X.iloc[:, 0].max()
    y_min = X.iloc[:, 1].min()
    y_max = X.iloc[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    pre_z = svc.predict(np.c_[XX.ravel(), YY.ravel()])

    Z = pre_z.reshape(XX.shape)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'])

    plt.pcolormesh(XX, YY, Z , cmap=plt.cm.Paired)
    plt.title(param)
    plt.show()

##############################################################

#K means Clustering

##dff= dataframe con los features
##dfr= dataframe con la response variable

def alg_kmeans(dff,n) :
    clusters = KMeans(n_clusters=n).fit_predict(dff)
    sns.scatterplot(dff[0], df[1], hue=clusters)


##########################################################

##Expectation-maximization Clustering

#dff= dataframe con los features ej. cards_features[['CA', 'CH']])

def alg_emc(dff,):

    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()

        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs), )
    def plot_gmm(gmm, X, label=True, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2, vmin=0, vmax=.2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2).set_ylim(top=.2, bottom = 0, vmin=0, vmax=.2)

        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)


    gmm = GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(dff)

    plot_gmm(gmm, np.array(dff)


#####################################################################

##Meanshift clusterization

def alg_meanshift(feat,x,y):
    bandwidth = estimate_bandwidth(feat, quantile=0.8)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    ms.fit(feat)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)


    centers_df = pd.DataFrame(cluster_centers).head()

    centers_df['x'] = centers_df[0]
    centers_df['y'] = centers_df[2]

    sns.scatterplot(x=x, y=y, data=cards_features, hue=labels)
    sns.scatterplot(x=x, y=y, data=centers_df, color='green', s=100)

#####################################################################

# Hierarchical Clustering

#df_feat = features del df

#dff = #dff= dataframe con los features ej. cards_features[['CA', 'CH']])

def alg_hc(dff,x,y,df_feat):
    for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=3)
    #t0 = time()
    clustering.fit(dff)

    sns.scatterplot(x='CA', y='CH', data=df_feat, hue=clustering.labels_)
    plt.figure()


######################################################################

#Linear regression

#col1= columna que va a predecir
#col2=columna a comparar

def alg_LinearR(df,col1,col2):
    y = pd.DataFrame(data=df, columns=[col1])
    X = new_imdb.drop(col1, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)
    X_train = sm.add_constant(X_train)
    simple_model = sm.OLS(y_train,X_train[[col2]])

    simple_result = simple_model.fit()

    X_test = sm.add_constant(X_test)
    y_pred_simple = simple_result.predict(X_test[[col2]])

    sns.scatterplot(x = X_test[col2], y = y_test.values.ravel())
    sns.lineplot(x = X_test[col2] , y = y_pred_simple)
    plt.title('Regression plot')


#######################################################################

#Linear Polynomial Regression

def alg_LinearPolynomial(df,col1,col2):
    y = pd.DataFrame(data=df, columns=[col1])
    X = new_imdb.drop(col1, axis=1)
    X_train, X_est, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)


    X_train = sm.add_constant(X_train)
    simple_model = sm.OLS(y_train,X_train[[col2, col1]])

    simple_result = simple_model.fit()


    X_test = sm.add_constant(X_test)
    y_pred_simple = simple_result.predict(X_test[[col2, col1]])
    sns.scatterplot(x = X_test[col1], y = y_test.values.ravel())
    sns.lineplot(x = X_test[col1] , y = y_pred_simple)



########################################################################

#Polynomial regression

#n= grados de la polinomial


def alg_poly(n,df,col1, col2)  :
        y = pd.DataFrame(data=df, columns=[col1])
        X = new_imdb.drop(col1, axis=1)
        X_train, X_est, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

        poly_reg = PolynomialFeatures(degree = n)
        X_poly_train = poly_reg.fit_transform(pd.DataFrame(X_train[col2]))
        X_poly_test = poly_reg.fit_transform(pd.DataFrame(X_test[col2]))
        poly_result = poly_reg.fit(X_poly_train, y_train)

        poly_model = LinearRegression()
        poly_result = poly_model.fit(X_poly_train, y_train)
        y_poly_pred = poly_model.predict(X_poly_test)

        sns.scatterplot(x = X_test[col2], y = y_test.values.ravel())
        sns.lineplot(x = X_test[col2] , y = y_pred_simple)
        sns.lineplot(x = X_test[col2] , y = y_poly_pred.ravel())

        poly_model = sm.OLS(y_train,X_poly_train)
        poly_result = poly_model.fit()

################################################################################


#LASSO


def alg_lasso(df, col1,col2):

    y = pd.DataFrame(data=df, columns=[col1])
    X = new_imdb.drop(col1, axis=1)
    X_train, X_est, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)


    lasso_model_0 = Lasso(alpha=0, fit_intercept=True)
    lasso_model_0.fit(X_train, y_train)



    lasso_model_10k = Lasso(alpha=10000, fit_intercept=True)
    lasso_model_10k.fit(X_train, y_train)
    lasso_10k_pred = lasso_model_10k.predict(X_test)


    sns.scatterplot(x = X_test[col2], y = y_test.values.ravel())
    sns.regplot(x = X_test[col2], y = lr_pred.ravel(), color='g')
    sns.regplot(x = X_test[col2], y = lasso_10k_pred.ravel(), color='purple')

    sns.scatterplot(x = X_test[col2], y = y_test.values.ravel())
    sns.regplot(x = X_test[col2], y = lr_pred.ravel(), color='g')
    sns.regplot(x = X_test[col2], y = lasso_10k_pred.ravel(), color='purple')


###################################################################################


#Ridge

#n= degrees

def  alg_ridge(n,df,col1,col2):
        y = pd.DataFrame(data=df, columns=[col1])
        X = new_imdb.drop(col1, axis=1)
        X_train, X_est, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

        poly_reg = PolynomialFeatures(degree = n)
        X_poly_train = poly_reg.fit_transform(pd.DataFrame(X_train))
        X_poly_test = poly_reg.fit_transform(pd.DataFrame(X_test))
        poly_result = poly_reg.fit(X_train, y_train)


        poly_model = LinearRegression()
        poly_result = poly_model.fit(X_poly_train, y_train)
        y_poly_pred = poly_model.predict(X_poly_test)

        lasso_poly_model_10k = Lasso(alpha=10, fit_intercept=True)
        lasso_poly_model_10k.fit(X_poly_train, y_train)
        lasso_poly_10k_pred = lasso_poly_model_10k.predict(X_poly_test)

        sns.scatterplot(x = X_test[col2], y = y_test.values.ravel())
        sns.regplot(x = X_test[col2], y = lr_pred.ravel(), color='g')
        sns.regplot(x = X_test[col2], y = y_poly_pred.ravel(), color='gray', order=3)
        sns.regplot(x = X_test[col2], y = lasso_poly_10k_pred.ravel(), color='purple', order=3)


###################################################################################################

## NN

## act= funcion de activacion para la capa oculta, ej. 'tanh', 'sigmoid' , 'Relu'
##opt=Optimizador de error, ej. 'sgd' ,'adam'
#Epoc= numeros de entrenamientos

def alg_nn(act,opt,epoc):
    model = keras.Sequential()

    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(64, activation=act))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer = opt,
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

    model_history = model1.fit(x = X_train,
                                y = y_train,
                                batch_size = 128,
                                epochs = epoc,
                                validation_split = 0.2,
                                shuffle=True)

    print(model_history.history.keys())

    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


##################################################################                                      
