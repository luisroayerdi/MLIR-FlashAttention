// RUN: attention-opt %s | attention-opt | FileCheck %s

module {
    // CHECK-LABEL: func @attention_types(%arg0: !attention.custom<"10">)
    func.func @attention_types(%arg0: !attention.custom<"10">) {
        return
    }
}
