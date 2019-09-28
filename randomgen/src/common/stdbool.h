#ifndef _RANDOMDGEN__STDBOOL_H
#define _RANDOMDGEN__STDBOOL_H

#if defined(_WIN32) && defined(_MSC_VER)
#ifndef __bool_true_false_are_defined
#define __bool_true_false_are_defined 1
typedef unsigned char bool;
#define false 0
#define true 1
#endif
#endif

#endif
