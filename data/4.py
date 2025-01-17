# import glob
# import tensorflow as tf
# import tensorflow_datasets as tfds

# dataset_dir = "/LOG/realman/LLM/RT1datasets"
# dataload = tfds.builder_from_directory(dataset_dir)
# dataset = dataload.as_data_source(split='train[:10%]')
# print(type(dataset))
# print(dataset)



import tensorflow as tf

# 创建数据集
dataset = tf.data.TFRecordDataset('/LOG/realman/LLM/RT1datasets')

# feature_description = {

#     '''Traceback (most recent call last):
#         File "/LOG/realman/LLM/robotic-transformer-pytorch/data/4.py", line 19, in <module>
#             'steps': Dataset({
#         NameError: name 'Dataset' is not defined '''

#     'steps': Dataset({
#         'action': FeaturesDict({
#             'base_displacement_vector': Tensor(shape=(2,), dtype=float32),
#             'base_displacement_vertical_rotation': Tensor(shape=(1,), dtype=float32),
#             'gripper_closedness_action': Tensor(shape=(1,), dtype=float32),
#             'rotation_delta': Tensor(shape=(3,), dtype=float32),
#             'terminate_episode': Tensor(shape=(3,), dtype=int32),
#             'world_vector': Tensor(shape=(3,), dtype=float32),
#         }),
#         'info': FeaturesDict({
#             'discounted_return': float32,
#             'return': float32,
#         }),
#         'is_first': bool,
#         'is_last': bool,
#         'is_terminal': bool,
#         'num_steps': int32,
#         'observation': FeaturesDict({
#             'base_pose_tool_reached': Tensor(shape=(7,), dtype=float32),
#             'gripper_closed': Tensor(shape=(1,), dtype=float32),
#             'gripper_closedness_commanded': Tensor(shape=(1,), dtype=float32),
#             'height_to_bottom': Tensor(shape=(1,), dtype=float32),
#             'image': Image(shape=(256, 320, 3), dtype=uint8),
#             'natural_language_embedding': Tensor(shape=(512,), dtype=float32),
#             'natural_language_instruction': string,
#             'orientation_box': Tensor(shape=(2, 3), dtype=float32),
#             'orientation_start': Tensor(shape=(4,), dtype=float32),
#             'robot_orientation_positions_box': Tensor(shape=(3, 3), dtype=float32),
#             'rotation_delta_to_go': Tensor(shape=(3,), dtype=float32),
#             'src_rotation': Tensor(shape=(4,), dtype=float32),
#             'vector_to_go': Tensor(shape=(3,), dtype=float32),
#             'workspace_bounds': Tensor(shape=(3, 3), dtype=float32),
#         }),
#         'reward': Scalar(shape=(), dtype=float32),
#         'step_id': int32,
#     }),
# }

feature_description_1 = {
                                "num_steps": {
                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                    "tensor": {
                                        "shape": {},
                                        "dtype": "int32",
                                        "encoding": "none"
                                    }
                                },
                                "observation": {
                                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                    "featuresDict": {
                                        "features": {
                                            "gripper_closed": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "1"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "src_rotation": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "4"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "vector_to_go": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                },
                                                "description": "displacement from current end-effector position to target"
                                            },
                                            "rotation_delta_to_go": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                },
                                                "description": "rotational displacement from current orientation to target"
                                            },
                                            "base_pose_tool_reached": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "7"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                },
                                                "description": "end-effector base-relative position+quaternion pose"
                                            },
                                            "natural_language_embedding": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "512"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "gripper_closedness_commanded": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "1"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                },
                                                "description": "continuous gripper position"
                                            },
                                            "height_to_bottom": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "1"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                },
                                                "description": "height of end-effector from ground"
                                            },
                                            "natural_language_instruction": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {},
                                                    "dtype": "string",
                                                    "encoding": "none"
                                                }
                                            },
                                            "workspace_bounds": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "3",
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "orientation_box": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "2",
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "robot_orientation_positions_box": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "3",
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "image": {
                                                "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                "image": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "256",
                                                            "320",
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "uint8",
                                                    "encodingFormat": "jpeg"
                                                }
                                            },
                                            "orientation_start": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "4"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            }
                                        }
                                    }
                                },
                                "action": {
                                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                    "featuresDict": {
                                        "features": {
                                            "base_displacement_vertical_rotation": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "1"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "base_displacement_vector": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "2"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "terminate_episode": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "int32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "gripper_closedness_action": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "1"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                },
                                                "description": "continuous gripper position"
                                            },
                                            "rotation_delta": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                },
                                                "description": "rpy commanded orientation displacement, in base-relative frame"
                                            },
                                            "world_vector": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {
                                                        "dimensions": [
                                                            "3"
                                                        ]
                                                    },
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                },
                                                "description": "commanded end-effector displacement, in base-relative frame"
                                            }
                                        }
                                    }
                                },
                                "is_terminal": {
                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                    "tensor": {
                                        "shape": {},
                                        "dtype": "bool",
                                        "encoding": "none"
                                    }
                                },
                                "step_id": {
                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                    "tensor": {
                                        "shape": {},
                                        "dtype": "int32",
                                        "encoding": "none"
                                    }
                                },
                                "is_first": {
                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                    "tensor": {
                                        "shape": {},
                                        "dtype": "bool",
                                        "encoding": "none"
                                    }
                                },
                                "reward": {
                                    "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                    "tensor": {
                                        "shape": {},
                                        "dtype": "float32",
                                        "encoding": "none"
                                    }
                                },
                                "is_last": {
                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                    "tensor": {
                                        "shape": {},
                                        "dtype": "bool",
                                        "encoding": "none"
                                    }
                                },
                                "info": {
                                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                    "featuresDict": {
                                        "features": {
                                            "return": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {},
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            },
                                            "discounted_return": {
                                                "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                "tensor": {
                                                    "shape": {},
                                                    "dtype": "float32",
                                                    "encoding": "none"
                                                }
                                            }
                                        }
                                    }
                                }
                            }

# 定义解析函数
def parse_function(example_proto):
    example = tf.io.parse_single_sequence_example(example_proto, feature_description_1)
    # example = tf.io.parse_single_example(example_proto, feature_description_1)
    steps = example['steps']
    # action = steps['action']
    # reward = example['reward']
    # next_state = example['next_state']
    # done = example['done']
    return steps

# 对数据集进行解析并预处理
dataset = dataset.map(parse_function)

print(dataset)
print(type(dataset))
# 打印数据集中的第一项
for steps in dataset.take(1):
    print('steps', steps)
    print('actions:', action)