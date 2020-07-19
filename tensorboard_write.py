from tensorboardX import SummaryWriter


def write_tensorboard(env_name, date, iter, value_loss, action_loss, bc_loss,
                        dist_entropy, dis_loss, gail_loss, grad_loss, ib_loss, task_loss, pos_loss):
    log_dir = './runs/'+ env_name + '_' + date

    summary = SummaryWriter(log_dir)

    summary.add_scalar('loss/value_loss', value_loss, iter)
    summary.add_scalar('loss/action_loss', action_loss, iter)
    summary.add_scalar('loss/bc_loss', bc_loss, iter)
    summary.add_scalar('loss/dist_entropy', dist_entropy, iter)
    summary.add_scalar('loss/dis_loss', dis_loss, iter)
    summary.add_scalar('loss/gail_loss', gail_loss, iter)
    summary.add_scalar('loss/grad_loss', grad_loss, iter)
    summary.add_scalar('loss/ib_loss', ib_loss, iter)
    summary.add_scalar('loss/task_loss', task_loss, iter)
    summary.add_scalar('loss/pos_loss', pos_loss, iter)


# def write_results(env_name, date, iter):
#     log_dir = './runs' + + 'results_' + env_name + '_' + date
#
#     summary = SummaryWriter(log_dir)
#
#