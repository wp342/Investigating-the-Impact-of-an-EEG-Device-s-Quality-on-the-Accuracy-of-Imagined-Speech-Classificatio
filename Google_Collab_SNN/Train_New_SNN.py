from Build_Siamese_NN import make_siamese_model
import tensorflow as tf
import os
import numpy as np

# define optimisers, variables, loss function and initialise siamese model
m = 0.5


def custom_contrastive_loss(y, yhat):
    max = tf.maximum((m - yhat), 0)
    max_sq = tf.square(max)
    yhat_sq = tf.square(yhat)
    multy1 = y * yhat_sq  # (1 / 2) *
    multy2 = (1 - y) * max_sq  # +(1 / 2) *
    return tf.add(multy1, multy2)


@tf.function
def train_step(batch, siamese_model, opt):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        X = [X[0], X[1]]
        # Get label
        y = batch[2]  # [0][0][0][0]

        # Forward pass
        yhat, embedding1, embedding2 = siamese_model(X, training=True)
        # Calculate loss
        # bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = custom_contrastive_loss(y, yhat)  # bce([[y]], yhat)
    # print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss, y, yhat, embedding1, embedding2


def train(inp1, inp2, inp3, data, all_epochs, task, folder, batch_size, validation_interval, save_per_epoch=20,
          model_fold=0, siamese_model=None, ):
    model_dir = 'SNN_Models/' + folder + '/' + task + '_normalised_CSP_3_fold/ '  # 'D:/MSc_Software_Systems/Research Project/FEIS/code_classification/SNN_Models/' + folder + '/' + task + '_CSP/ '
    # tf.keras.backend.clear_session()
    opt = tf.keras.optimizers.Adam(1e-4)
    # initialise model and checkpoints
    if siamese_model is None:
        siamese_model = make_siamese_model(inp1, inp2, inp3)
    # gradient_tape = tf.GradientTape()
    tf.keras.backend.clear_session()

    # train_size = int(len(data[0]) * .85)
    # train_data = (data[0][0:train_size], data[1][0:train_size], data[2][0:train_size])
    # val_data = (data[0][train_size:], data[1][train_size:], data[2][train_size:])
    
    train_data = (data[0], data[1], data[2])
    train_data = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(train_data[0]), tf.data.Dataset.
                                      from_tensor_slices(train_data[1]), tf.data.Dataset.
                                      from_tensor_slices(train_data[2])))
    extra = 0
    for index, epochs in enumerate(all_epochs):
        data = train_data.batch(batch_size[index])
        data = data.prefetch(int(batch_size[index] / 2))

        print("Beginning Training with a Batch size of " + str(batch_size[index]) + " and " + str(epochs), " epochs")

        # Loop through epochs
        for epoch in range(1, epochs + 1):
            print('\n Epoch {}/{}'.format(epoch, epochs))
            progbar = tf.keras.utils.Progbar(len(data))

            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss, y, yhat, embedding1, embedding2 = train_step(batch, siamese_model, opt)
                print()
                # print('embedding 1: ' + str(embedding1))
                # print('embedding 2: ' + str(embedding2))
                #print('Distances: ' + str(yhat))
                #print('LOSS: ' + str(loss))
                # print('Y: ' + str(y))
                progbar.update(idx + 1)

            # if epoch % validation_interval == 0:
            #     print("Running validation......")
            #     siamese_model.compile(optimizer="Adam", loss=custom_contrastive_loss)
            #     predictions, _, _ = siamese_model.predict([val_data[0], val_data[1]])
            #     average = np.mean(predictions)
            #     counter = 0
            #     for index, predict in enumerate(predictions):
            #         if predict > average:
            #             if val_data[2][index] == 0:
            #                 counter = counter + 1
            #         else:
            #             if val_data[2][index] == 1:
            #                 counter = counter + 1
            #     print("Validation score is: " + str(counter / len(predictions)))
            
            if batch_size[index] < batch_size[0]:
                extra = all_epochs[0]
                

            # saving the model
            if epoch % save_per_epoch == 0:
                # siamese_model.save(model_dir + 'SNN_model_' + str(epoch))
                siamese_model.save(model_dir + 'SNN_model_' + str(epoch + extra) + '_' + str(model_fold) + '.h5')

    return siamese_model
    
def train_bc(inp1, inp2, inp3, data, all_epochs, task, folder, batch_size, validation_interval, save_per_epoch=20,
          model_fold=0, siamese_model=None, ):
    model_dir = 'SNN_Models/' + folder + '/' + task + '_normalised_CSP_3_fold/ '  # 'D:/MSc_Software_Systems/Research Project/FEIS/code_classification/SNN_Models/' + folder + '/' + task + '_CSP/ '
    # tf.keras.backend.clear_session()
    opt = tf.keras.optimizers.Adam(1e-4)
    # initialise model and checkpoints
    if siamese_model is None:
        siamese_model = make_siamese_model(inp1, inp2, inp3)
    # gradient_tape = tf.GradientTape()
    tf.keras.backend.clear_session()
    train_data = []
    for d in data:
        train_data.append(tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(d[0]), tf.data.Dataset.
                                          from_tensor_slices(d[1]), tf.data.Dataset.
                                          from_tensor_slices(d[2]))))

    extra = 0
    for index, epochs in enumerate(all_epochs):


        data = train_data[index].batch(batch_size[index])
        data = data.prefetch(int(batch_size[index] / 2))
        if index == 1:
            for i in range(7):
                siamese_model.layers[2].layers[i].trainable = False
        print("Beginning Training with a Batch size of " + str(batch_size[index]) + " and " + str(epochs), " epochs")

        # Loop through epochs
        for epoch in range(1, epochs + 1):
            print('\n Epoch {}/{}'.format(epoch, epochs))
            progbar = tf.keras.utils.Progbar(len(data))

            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss, y, yhat, embedding1, embedding2 = train_step(batch, siamese_model, opt)
                print()
                # print('embedding 1: ' + str(embedding1))
                # print('embedding 2: ' + str(embedding2))
                print('Distances: ' + str(yhat))
                print('LOSS: ' + str(loss))
                # print('Y: ' + str(y))
                progbar.update(idx + 1)
            
            if batch_size[index] < batch_size[0]:
                extra = all_epochs[0]
            
            # saving the model
            if epoch % save_per_epoch == 0:
                # siamese_model.save(model_dir + 'SNN_model_' + str(epoch))
                siamese_model.save(model_dir + 'SNN_model_' + str(epoch + extra) + '_' + str(model_fold) + '.h5')

    return siamese_model