import tensorflow as tf


def evaluate(model, run_paths, ds_test):
    # 检查点路径
    checkpoint_dir = run_paths['path_ckpts_train']

    # 查找最新的检查点文件
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Loading model checkpoint: {latest_checkpoint}")
        ckpt = tf.train.Checkpoint(model=model)
        status = ckpt.restore(latest_checkpoint)
        status.expect_partial()  # 忽略未找到的变量
    else:
        print("No checkpoint found.")
        return

    # 准备评估指标
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    # Evaluation step
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)


    # 迭代测试数据集
    for test_images, test_labels in ds_test:
        test_step(test_images, test_labels)

    # 打印结果
    print(f"Test Loss: {test_loss.result()}")
    print(f"Test Accuracy: {test_accuracy.result() * 100}")


    # 返回评估指标
    return test_loss.result().numpy(), test_accuracy.result().numpy()