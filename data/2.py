
# import tensorflow as tf # 定义解析函数 
# def parse_function(serialized_example): 
#     feature_description = {
#     "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
#     "featuresDict": {
#         "features": {
#             "steps": {
#                 "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
#                 "sequence": {
#                     "feature": {
#                         "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
#                         "featuresDict": {
#                             "features": {
#                                 "num_steps": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                     "tensor": {
#                                         "shape": {},
#                                         "dtype": "int32",
#                                         "encoding": "none"
#                                     }
#                                 },
#                                 "observation": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
#                                     "featuresDict": {
#                                         "features": {
#                                             "gripper_closed": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "1"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "src_rotation": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "4"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "vector_to_go": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 },
#                                                 "description": "displacement from current end-effector position to target"
#                                             },
#                                             "rotation_delta_to_go": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 },
#                                                 "description": "rotational displacement from current orientation to target"
#                                             },
#                                             "base_pose_tool_reached": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "7"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 },
#                                                 "description": "end-effector base-relative position+quaternion pose"
#                                             },
#                                             "natural_language_embedding": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "512"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "gripper_closedness_commanded": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "1"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 },
#                                                 "description": "continuous gripper position"
#                                             },
#                                             "height_to_bottom": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "1"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 },
#                                                 "description": "height of end-effector from ground"
#                                             },
#                                             "natural_language_instruction": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {},
#                                                     "dtype": "string",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "workspace_bounds": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "3",
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "orientation_box": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "2",
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "robot_orientation_positions_box": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "3",
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "image": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
#                                                 "image": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "256",
#                                                             "320",
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "uint8",
#                                                     "encodingFormat": "jpeg"
#                                                 }
#                                             },
#                                             "orientation_start": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "4"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             }
#                                         }
#                                     }
#                                 },
#                                 "action": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
#                                     "featuresDict": {
#                                         "features": {
#                                             "base_displacement_vertical_rotation": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "1"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "base_displacement_vector": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "2"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "terminate_episode": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "int32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "gripper_closedness_action": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "1"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 },
#                                                 "description": "continuous gripper position"
#                                             },
#                                             "rotation_delta": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 },
#                                                 "description": "rpy commanded orientation displacement, in base-relative frame"
#                                             },
#                                             "world_vector": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {
#                                                         "dimensions": [
#                                                             "3"
#                                                         ]
#                                                     },
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 },
#                                                 "description": "commanded end-effector displacement, in base-relative frame"
#                                             }
#                                         }
#                                     }
#                                 },
#                                 "is_terminal": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                     "tensor": {
#                                         "shape": {},
#                                         "dtype": "bool",
#                                         "encoding": "none"
#                                     }
#                                 },
#                                 "step_id": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                     "tensor": {
#                                         "shape": {},
#                                         "dtype": "int32",
#                                         "encoding": "none"
#                                     }
#                                 },
#                                 "is_first": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                     "tensor": {
#                                         "shape": {},
#                                         "dtype": "bool",
#                                         "encoding": "none"
#                                     }
#                                 },
#                                 "reward": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
#                                     "tensor": {
#                                         "shape": {},
#                                         "dtype": "float32",
#                                         "encoding": "none"
#                                     }
#                                 },
#                                 "is_last": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                     "tensor": {
#                                         "shape": {},
#                                         "dtype": "bool",
#                                         "encoding": "none"
#                                     }
#                                 },
#                                 "info": {
#                                     "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
#                                     "featuresDict": {
#                                         "features": {
#                                             "return": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {},
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             },
#                                             "discounted_return": {
#                                                 "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
#                                                 "tensor": {
#                                                     "shape": {},
#                                                     "dtype": "float32",
#                                                     "encoding": "none"
#                                                 }
#                                             }
#                                         }
#                                     }
#                                 }
#                             }
#                         }
#                     },
#                     "length": "-1"
#                 }
#             }
#         }
#     }
# } 
#     example = tf.io.parse_single_example(serialized_example, feature_description) 
#     # 返回解析后的特征 
#     return example
#     # 创建TFRecordDataset对象 

# filename = '/LOG/realman/LLM/RT1datasets/fractal_fractal_20220817_data_traj_transform_rt_1_without_filters_disable_episode_padding_seq_length_6_no_preprocessor-train.array_record-00000-of-01024'

# dataset = tf.data.TFRecordDataset(‘.array_record’) 
# # 对每个样本应用解析函数 
# dataset = dataset.map(parse_function) # 遍历数据集 
# for features in dataset: 
#     print(features)

import tensorflow as tf

# 加载数据集，并转换成tfrecord格式
dataset = tf.data.Dataset.from_tensor_slices((features_array, targets_array))
tfrec = 'my_data.tfrecord'
writer = tf.data.experimental.TFRecordWriter(tfrec)
writer.write(dataset)

# 读取保存为TFRecord文件的数据集
def parse_fn(example_proto):
    features = {
    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
    "featuresDict": {
        "features": {
            "steps": {
                "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                "sequence": {
                    "feature": {
                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                        "featuresDict": {
                            "features": {
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
                        }
                    },
                    "length": "-1"
                }
            }
        }
    }
}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features['features'], parsed_features['targets']

# 使用tf.data.TFRecordDataset函数读取TFRecord文件
filenames = [tfrec]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parse_fn)

# 设置batch size，并打乱数据集
batch_size = 10
dataset = dataset.batch(batch_size, drop_remainder=True).shuffle(buffer_size=100)
