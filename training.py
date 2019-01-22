import argparse

parser = argparse.ArgumentParser(description='CMPT726 Project')
mutually_exclusive_model_options = parser.add_mutually_exclusive_group(required=True)
mutually_exclusive_model_options.add_argument('--model_series', metavar='model', type=int,
	choices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], help='Choose model series'
)
mutually_exclusive_model_options.add_argument('--resume_model', metavar='model file', type=str,
	help='Saved model to resume training from'
)
mutually_exclusive_data_options = parser.add_mutually_exclusive_group(required=True)
mutually_exclusive_data_options.add_argument('--rescale', action='store_true', help='Rescale input')
mutually_exclusive_data_options.add_argument('--normalize', action='store_true', help='Normalize input')

parser.add_argument('--save_dest', metavar='dest', type=str, required=True, help='Save model destination')
parser.add_argument('--batch_size', metavar='bs', type=int, required=False, default=8, help='Batch size')
parser.add_argument('--dataset', type=str, required=False, default='cifar100', choices=['cifar10', 'cifar100'], help='Dataset to work on')
parser.add_argument('--epochs', type=int, required=False, default=50, help='Epochs')
parser.add_argument('--resume_from_epoch', type=int, required=False, default=None, help='Resume from this epoch (0 index). Required for resumption')
parser.add_argument('--lr', metavar='lr', type=float, required=False, default=0.0001, help='Learning rate')
parser.add_argument('--super_loss_weight', metavar='sup_w', type=float, required=False, default=0.2, help='Loss weight for superclass')
parser.add_argument('--sub_loss_weight', metavar='sub_w', type=float, required=False, default=0.8, help='Loss weight for subclass')
parser.add_argument('--gray', required=False, action='store_true', help='Turn images to RGB first or not')
parser.add_argument('--init', required=False, type=str, default='he_normal', choices=['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'], help='Kernel initializers')
parser.add_argument('--loss', required=False, type=str, choices=['cc', 'margin', 'seg_margin'], default='margin', help='Loss function to use')
parser.add_argument('--margin_downweight', required=False, type=float, default=0.5, help='Margin loss downweight - 0 <= downweight <= 1')
parser.add_argument('--pos_margin', required=False, type=float, default=0.9, help='Positive margin - 0 <= pos_margin <= 1')
parser.add_argument('--neg_margin', required=False, type=float, default=0.1, help='Negative margin - 0 <= neg_margin <= 1')
parser.add_argument('--val_split', type=float, required=False, default=0.1, help='Validation split')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='Optimizer to use (hyperparameters are specified by default)')
parser.add_argument('-tb', '--tensorboard', required=False, action='store_true', help='Use tensorboard or not')
parser.add_argument('--tb_dir', type=str, required=False, default='./tensorboard', help='Tensorboard directory (only applies if -tb is given)')
parser.add_argument('--tb_rate', type=int, required=False, default=1000, help='Tensorboard update rate')
parser.add_argument('--checkpoint', required=False, action='store_true', help='Whether to checkpoint model or not')
parser.add_argument('--checkpoint_file', required=False, default='model.{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5', help='File to checkpoint to')
parser.add_argument('--workers', metavar='w', type=int, required=False, default=1, help='Number of workers')

# parse sys.argv
args = parser.parse_args()

if args.resume_model is not None and args.resume_from_epoch is None:
	parser.error('--resume_model requires --resume_from_epoch to be stated')

import utils
import models
from keras import callbacks

'''Point of Comparison for Image Augmentation'''
# configure batch size and retrieve one batch of images

if args.model_series is not None:
	if args.model_series == 1:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelOne, args, coarse_too=True)
	elif args.model_series == 2:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelTwo, args)
	elif args.model_series == 3:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelThree, args, coarse_too=True)
	elif args.model_series == 4:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelFour, args, coarse_too=True)
	elif args.model_series == 5:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelFive, args)
	elif args.model_series == 6:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelSix, args)
	elif args.model_series == 7:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelSeven, args)
	elif args.model_series == 8:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelEight, args)
	elif args.model_series == 9:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelNine, args)
	elif args.model_series == 10:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelTen, args)
	elif args.model_series == 11:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelEleven, args)
	elif args.model_series == 12:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelTwelve, args)
	elif args.model_series == 13:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelThirteen, args)
	elif args.model_series == 14:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelFourteen, args)
	elif args.model_series == 15:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelFifteen, args)
	elif args.model_series == 16:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelSixteen, args)
	elif args.model_series == 17:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelSeventeen, args)
	elif args.model_series == 18:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelEighteen, args)
	elif args.model_series == 19:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelNineteen, args)
	elif args.model_series == 20:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelTwenty, args)
	elif args.model_series == 21:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelTwentyOne, args)	
	elif args.model_series == 22:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelTwentyTwo, args)
	elif args.model_series == 23:
		dataset, gen, model = utils.prepare_for_model(models.TrialModelTwentyThree, args)			
else:
	dataset, gen, model = utils.prepare_for_model(None, args)

model.summary(line_length=150)

try:
	cbs = []

	if args.tensorboard:
		print('Will record for tensorboard to {tb_dir}'.format(tb_dir=args.tb_dir))
		tb = callbacks.TensorBoard(log_dir=args.tb_dir, write_graph=False, write_grads=True,
					histogram_freq=1, batch_size=args.batch_size, update_freq=args.tb_rate)
		cbs.append(tb)
	if args.checkpoint:
		print('Will checkpoint best model... checkpoint model name format is {cpfile}'.format(cpfile=args.checkpoint_file))
		checkpointer = callbacks.ModelCheckpoint(args.checkpoint_file, monitor='val_categorical_accuracy',
					verbose=1, save_best_only=True, mode='max')
		cbs.append(checkpointer)
	
	if dataset['y_coarse']['val'] is not None:
		validation_data=(dataset['X']['val'], [dataset['y_coarse']['val'], dataset['y_fine']['val']])
	else:
		validation_data=(dataset['X']['val'], dataset['y_fine']['val'])

	initial_epoch = args.resume_from_epoch if args.resume_from_epoch is not None else 0

	train_history = model.fit_generator(gen, epochs=args.epochs,
		steps_per_epoch=dataset['X']['train'].shape[0] // args.batch_size,
		validation_data=validation_data, verbose=1, max_queue_size=10,
		workers=args.workers, callbacks=cbs, initial_epoch=initial_epoch)
except KeyboardInterrupt:
	print('Keyboard interrupted during training...')
finally:
	model.save(args.save_dest)