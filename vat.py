# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Activation, Dropout, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Cropping2D, GaussianNoise, Add
from keras.utils import np_utils
from keras.models import Model
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.optimizers import Adam

class VATWrapper:
    def __init__(self, eps=8.0, xi=1.e-6, ip=1, alpha=1., supervisedOnly=False):
        self.alpha = alpha
        self.eps = eps
        self.xi = xi
        self.ip = ip
        self.supervisedOnly = supervisedOnly
        return

    def setModel(self,model):
        self.model = model

        self.inX  = model.inputs[0]
        self.inputLorUL = Input( (1,), name="LorUL" )
        self.outY = model.outputs[0]

        self.newModel = Model(inputs=[self.inX,self.inputLorUL],outputs=[self.outY])

        return self.newModel

    def vat_loss(self, eps, xi, ip):
        normal_outputs = K.stop_gradient(self.outY)
        d_list = K.random_normal(K.shape(self.inX))
        d_list = self.normalize_vector(d_list)

        for _ in range(ip):
            new_inputs = K.stop_gradient(self.inX) + d_list*xi
            new_outputs = self.model(new_inputs)
            kld = self.kld(normal_outputs, new_outputs)
            d_list = K.gradients(kld, d_list)[0] # gradientsの最初のターム(loss)が、バッチごとで良いのかどうかは要確認
            d_list = self.normalize_vector(d_list)
            d_list = K.stop_gradient(d_list)

        # ここから先が微分つながっていてほしい。
        new_inputs = self.inX + K.stop_gradient(d_list) * eps
        y_perturbations = self.model(new_inputs)
        kld = self.kld(normal_outputs, y_perturbations) # バッチごとにしないと行けない
        return kld

    def getSoftMaxloss(self):
        weights = K.reshape(self.inputLorUL,(K.shape(self.inputLorUL)[0],))
        def SoftmaxLoss(y_true, y_pred):
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            loss  = y_true * K.log(y_pred)
            loss  = -K.sum(loss, axis=-1)
            loss  = loss * weights
            return loss
        return SoftmaxLoss

    def getVATloss(self):
        weights = K.reshape(self.inputLorUL,(K.shape(self.inputLorUL)[0],))
        def VATLoss(y_true, y_pred):
            vatloss = self.vat_loss(self.eps,self.xi,self.ip)
            #vatloss = K.reshape(vatloss,(K.shape(vatloss)[0],)) * (1.-weights)
            vatloss = K.reshape(vatloss,(K.shape(vatloss)[0],))
            smloss  = self.getSoftMaxloss()(y_true, y_pred) * weights
            #loss = self.alpha * vat * (1.-weights) + smloss * weights
            #loss = self.alpha * vatloss + smloss
            loss = self.alpha * vatloss
            #loss = self.alpha * vatloss
            #loss = vatloss
            return loss
        return VATLoss

    def getLoss(self):
        def loss(y_true, y_pred):
            smloss = self.getSoftMaxloss()(y_true, y_pred)
            vatloss = self.getVATloss()(y_true, y_pred)
            loss = (smloss + vatloss) + 10.*(smloss-vatloss)*(smloss-vatloss)
            return loss
        return loss

    def getVATaccu(self):
        weights = K.reshape(self.inputLorUL,(K.shape(self.inputLorUL)[0],))
        def accuracy(y_true,y_pred):
            y_true = K.argmax(y_true, axis=-1)
            y_pred = K.argmax(y_pred, axis=-1)
            total  = K.cast(K.equal(y_true,y_pred),K.floatx())
            total *= weights
            total  = K.sum(total)

            count = K.sum(weights)
            count = K.maximum(count,1) # fail safe. 
            return total/count
        return accuracy

    @staticmethod
    def normalize_vector(x):
        z = K.sum(K.batch_flatten(K.square(x)), axis=1)
        while K.ndim(z) < K.ndim(x):
            z = K.expand_dims(z, axis=-1)
        return x / (K.sqrt(z) + K.epsilon())

    @staticmethod
    def kld(p, q):
        v = p * (K.log(p + K.epsilon()) - K.log(q + K.epsilon()))
        v = K.clip(v,min_value=0.,max_value=1e10)
        return K.sum(K.batch_flatten(v), axis=1, keepdims=True)

if __name__=="__main__":

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    #
    percent = 100./X_train.shape[0]
    NtrainPerClass = 10
    print "N_train per class = %d"%(NtrainPerClass)
    p = float(NtrainPerClass)/X_train.shape[0] * 10
    u_train = np.random.choice([0.,1.],y_train.shape[0],p=[1.-p,p])# np.array([1.]*y_train.shape[0])
    u_test  = np.array([1.]*y_test.shape[0])

    # convert class vectors to binary class matrices
    y_train[u_train==0.] = 0
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # define model
    inX = Input(input_shape)
    h = inX
    h = Conv2D(32, (3,3), padding='valid', activation="relu")(h)
    h = Conv2D(32, (3,3), activation="relu")(h)
    h = MaxPooling2D(pool_size=(2,2))(h)
    h = Flatten()(h)
    h = Dense(128, activation="relu")(h)
    h = Dense(10, activation='softmax')(h)
    outY = h

    classifier = Model(inX,outY)

    # VAT model definition
    vatWrapper = VATWrapper(eps=8.0,xi=1e-6)
    vatModel   = vatWrapper.setModel(classifier)

    #model.compile(loss=vatLoss, optimizer='adadelta', metrics=[vatAccu])
    optimizer = Adam(lr=2e-4)
    vatModel.compile(loss=vatWrapper.getSoftMaxloss(), optimizer=optimizer, metrics=[vatWrapper.getVATaccu()])
    vatModel.summary()
    vatModel.fit([X_train,u_train], y_train, batch_size=128, epochs=100000000,verbose=1, validation_data=([X_test,u_test], y_test))
